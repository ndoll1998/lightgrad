import sys
sys.path.insert(0, "../")
import lightgrad as light
import lightgrad.nn as nn
from lightgrad.autograd.utils.profiler import Profiler

""" BERT Model """
import math, json
import numpy as np

# fast gelu
gelu = lambda x: 0.5 * x * (1.0 + (x *  0.7978845608 * (1.0 + 0.044715 * x * x)).tanh())

class Embedding(nn.Module):
    def __init__(self, embedding_dim:int, vocab_size:int):
        nn.Module.__init__(self)
        self.d, self.n = embedding_dim, vocab_size
        self.weight = light.xavier((vocab_size, embedding_dim))
    def forward(self, ids):
        # TODO: hax - we dont support this kind of indexing yet
        return self.weight.cpu()[ids.cpu()].opencl()
        
class BertEmbedding(nn.Module):
    def __init__(self,
        hidden_size:int,
        vocab_size:int,
        max_position_embeddings:int,
        type_vocab_size:int,
    ):
        nn.Module.__init__(self)
        # embeddings
        self.word_embeddings = Embedding(hidden_size, vocab_size)
        self.position_embeddings = Embedding(hidden_size, max_position_embeddings)
        self.token_type_embeddings = Embedding(hidden_size, type_vocab_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        # TODO:
        self.dropout = lambda x: x
    def forward(self, input_ids, token_type_ids=None):
        # build token type ids
        if token_type_ids is None:
            token_type_ids = input_ids.__class__.zeros(input_ids.shape, dtype=np.int32)
        # build position ids
        position_ids = np.arange(input_ids.shape[-1], dtype=np.int32)
        position_ids = input_ids.__class__.from_numpy(position_ids)
        # get embeddings
        input_embedd = self.word_embeddings(input_ids)
        position_embedd = self.position_embeddings(position_ids)
        type_embedd = self.token_type_embeddings(token_type_ids)
        # compute final embedding
        embedd = input_embedd + position_embedd + type_embedd
        return self.dropout(self.LayerNorm(embedd))

class BertSelfAttention(nn.Module):
    def __init__(self,
        hidden_size:int,
        num_attention_heads:int
    ):
        nn.Module.__init__(self)
        assert hidden_size % num_attention_heads == 0
        self.h = num_attention_heads
        self.d = hidden_size // num_attention_heads
        # key, value and query 
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        # TODO:
        self.dropout = lambda x: x
    def forward(self, hidden, attention_mask=None):
        # mixed query, key, value
        Q = self.query(hidden)
        K = self.key(hidden)
        V = self.value(hidden)
        # split attention heads
        b, s, _ = K.shape
        Q = Q.reshape(b, s, self.h, self.d).transpose(0, 2, 1, 3)
        K = K.reshape(b, s, self.h, self.d).transpose(0, 2, 3, 1)
        V = V.reshape(b, s, self.h, self.d).transpose(0, 2, 1, 3)
        # scaled dot product attention
        scores = Q @ K / math.sqrt(self.d)
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(1, 1, *attention_mask.shape)
            attention_mask = (1.0 - attention_mask) * -10000.0
            scores = scores + attention_mask.detach()
        scores = self.dropout(scores.softmax(axis=-1))
        context = (scores @ V).transpose(0, 2, 1, 3)
        # concatenate attention heads
        context = context.reshape(b, s, self.h * self.d)
        return context, scores

class BertAttention(nn.Module):
    def __init__(self,
        hidden_size:int, 
        num_attention_heads:int
    ):
        nn.Module.__init__(self)
        # self attention
        self.self = BertSelfAttention(hidden_size, num_attention_heads)
        self.output = nn.Module()
        self.output.dense = nn.Linear(hidden_size, hidden_size)
        self.output.LayerNorm = nn.LayerNorm(hidden_size)
        # TODO
        self.output.dropout = lambda x: x
    def forward(self, hidden_in, attention_mask=None):
        hidden, attentions = self.self(hidden_in, attention_mask=attention_mask)
        hidden = self.output.dense(hidden)
        hidden = self.output.dropout(hidden)
        hidden = self.output.LayerNorm(hidden + hidden_in)
        return hidden, attentions

class BertLayer(nn.Module):
    def __init__(self,
        hidden_size:int,
        intermediate_size:int,
        num_attention_heads:int
    ):
        nn.Module.__init__(self)
        self.attention = BertAttention(hidden_size, num_attention_heads)
        self.intermediate = nn.Module()
        self.intermediate.dense = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Module()
        self.output.dense = nn.Linear(intermediate_size, hidden_size)
        self.output.LayerNorm = nn.LayerNorm(hidden_size)
        # TODO:
        self.output.dropout = lambda x: x
    def mlp(self, x):
        y = self.intermediate.dense(x)
        y = self.output.dense(gelu(y))
        return y
    def forward(self, hidden, attention_mask=None):
        # apply self attention
        hidden, attentions = self.attention(hidden, attention_mask)
        hidden = hidden + self.mlp(hidden)
        hidden = self.output.LayerNorm(self.output.dropout(hidden))
        return hidden, attentions

class BertModel(nn.Module):
    def __init__(self, 
        # encoder
        hidden_size:int,
        intermediate_size:int,
        num_hidden_layers:int,
        num_attention_heads:int,
        # embeddings
        vocab_size:int,
        max_position_embeddings:int,
        type_vocab_size:int,
        # dropout probs
        attention_probs_dropout_prob:float,
        hidden_dropout_prob:float
    ):
        nn.Module.__init__(self)
        # embeddings and encoder
        self.embeddings = BertEmbedding(hidden_size, vocab_size, max_position_embeddings, type_vocab_size)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(*[
            BertLayer(hidden_size, intermediate_size, num_attention_heads)
            for _ in range(num_hidden_layers)
        ])
    def forward(self, 
        input_ids, 
        attention_mask=None,
        token_type_ids=None
    ):
        hidden = self.embeddings(input_ids, token_type_ids=token_type_ids)
        for l in self.encoder.layer:
            hidden, _ = l(hidden, attention_mask=attention_mask)
        return hidden

class BertForMaskedLM(nn.Module):
    def __init__(self,
        # encoder
        hidden_size:int,
        intermediate_size:int,
        num_hidden_layers:int,
        num_attention_heads:int,
        # embeddings
        vocab_size:int,
        max_position_embeddings:int,
        type_vocab_size:int,
        # dropout probs
        attention_probs_dropout_prob:float,
        hidden_dropout_prob:float,
        **kwargs
    ):
        nn.Module.__init__(self)
        # bert model
        self.bert = BertModel(
            # encoder
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            # embeddings
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            # dropout probs
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob
        )
        # masked language modeling head
        self.cls = nn.Module()
        self.cls.predictions = nn.Module()
        # transform
        self.cls.predictions.transform = nn.Module()
        self.cls.predictions.transform.dense = nn.Linear(hidden_size, hidden_size)
        self.cls.predictions.transform.LayerNorm = nn.LayerNorm(hidden_size)
        # decoder
        self.cls.predictions.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.cls.predictions.bias = light.zeros(vocab_size)
    def forward(self, 
        input_ids, 
        attention_mask=None,
        token_type_ids=None
    ):
        # apply bert
        h = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        # transform
        h = self.cls.predictions.transform.dense(h)
        h = gelu(h)
        h = self.cls.predictions.transform.LayerNorm(h)
        # decoder
        y = self.cls.predictions.decoder(h) + self.cls.predictions.bias
        return y
    @staticmethod
    def from_pretrained(pretrained_name):
        from lightgrad.utils import fetch, load_torch_state_dict
        # fetch
        config_bytes = fetch(f"https://huggingface.co/{pretrained_name}/resolve/main/config.json")
        params_bytes = fetch(f"https://huggingface.co/{pretrained_name}/resolve/main/pytorch_model.bin")
        # load
        config = json.loads(config_bytes)
        params = load_torch_state_dict(params_bytes)
        # support older pytorch versions
        params = {n.replace('LayerNorm.gamma', 'LayerNorm.weight').replace('LayerNorm.beta', 'LayerNorm.bias'): p for n, p in params.items()}
        # create model
        model = BertForMaskedLM(**config)
        model.load_parameters(params, separator='.')
        return model

""" Bert Tokenizer """
import unicodedata
from collections import OrderedDict

class BertTokenizer(object):

    def __init__(self, 
        vocab:list,
        do_lower_case:bool =True,
        unk_token:str ="[UNK]",
        mask_token:str ="[MASK]",
        sep_token:str ='[SEP]'
    ):
        self.vocab = OrderedDict({word: i for i, word in enumerate(vocab)})
        self.do_lower_case = do_lower_case
        # special tokens
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.sep_token = sep_token

    def is_punct(self, char):
        cp, cat = ord(char), unicodedata.category(char)
        punct = (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)
        return punct or cat.startswith("P")

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[t] for t in tokens]

    def convert_ids_to_tokens(self, ids):
        self.words = list(self.vocab.keys())
        return [self.words[i] for i in ids]

    def tokenize(self, text:str):
        # tokenize
        tokens = []
        for whitespace_token in text.strip().split():
            # special tokens
            if whitespace_token in [self.unk_token, self.mask_token, self.sep_token]:
                tokens.append(whitespace_token)
                continue
            for token in self.punctuation_tokenize(whitespace_token):
                # word is too long
                if len(token) > 100:
                    output_tokens.append(self.unk_token)
                    continue
                # wordpiece tokenization
                token = token.lower() if self.do_lower_case else token
                tokens.extend(self.wordpiece_tokenize(token))
        return tokens

    def punctuation_tokenize(self, text):
        last_split = 0
        for i, c in enumerate(text):
            if self.is_punct(c):
                yield text[last_split:i]
                yield text[i:i+1]
                last_split = i+1
        yield text[last_split:]

    def wordpiece_tokenize(self, text):
        wordpieces = []
        is_bad, start = False, 0
        while start < len(text):
            cur_substr, end = None, len(text)
            while start < end:
                substr = ('##' if start > 0 else '') + text[start:end]
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                is_bad = True
                break
            wordpieces.append(cur_substr)
            start = end
        return [self.unk_token] if is_bad else wordpieces

    @staticmethod
    def from_pretrained(pretrained_name, **kwargs):
        from lightgrad.utils import fetch
        # fetch vocab
        vocab_bytes = fetch(f"https://huggingface.co/{pretrained_name}/resolve/main/vocab.txt")
        vocab = vocab_bytes.decode('utf-8').strip().split('\n')
        # create tokenizer
        return BertTokenizer(vocab, **kwargs)


if __name__ == '__main__':

    # model and sample text - currently only one mask word allowed
    k = 10
    pretrained_name = 'bert-base-uncased'
    text = "Alexander's legacy includes the [MASK] diffusion and syncretism which his conquests engendered, such as Greco-Buddhism."    # cultural
    # create model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_name)
    model = BertForMaskedLM.from_pretrained(pretrained_name).map_params(lambda p: p.opencl())
    # tokenize and get mask token index
    tokens = tokenizer.tokenize(text)
    mask_i = tokens.index('[MASK]')
    # predict
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = light.from_numpy(np.asarray([input_ids], dtype=np.int32))
    with light.no_grad(), Profiler() as p:
            lm_logits = model.forward(input_ids).numpy()
            p.print()
    lm_logits = lm_logits[0, mask_i, :]
    top_k = lm_logits.argsort()[-k:]
    # get tokens of top-k indices
    top_k_words = tokenizer.convert_ids_to_tokens(top_k.tolist()[::-1])

    print(text)
    print("Possible words:", top_k_words)

