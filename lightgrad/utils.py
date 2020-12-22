
def fetch(url):
    import requests, os, hashlib, tempfile
    fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp) and os.stat(fp).st_size > 0:
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        print("fetching %s" % url)
        dat = requests.get(url).content
        with open(fp+".tmp", "wb") as f:
            f.write(dat)
        os.rename(fp+".tmp", fp)
    return dat

def load_torch_state_dict(bytes_):
    import numpy as np
    import io, pickle, struct

    # convert it to a file
    fb0 = io.BytesIO(bytes_)
    # skip three junk pickles
    pickle.load(fb0)
    pickle.load(fb0)
    pickle.load(fb0)

    # pytorch data-storages
    torch_storages = {
        'FloatStorage': np.float32,
        'LongStorage': np.int64
    }

    key_prelookup = {}
    class TorchTensor:
        def __new__(cls, *args):
            ident, storage_type, obj_key, location, obj_size, view_metadata = args[0]
            assert ident == 'storage'
            tensor = np.zeros(obj_size, dtype=storage_type)
            key_prelookup[obj_key] = (storage_type, obj_size, tensor, args[2], args[3])
            return tensor

    class TorchUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # handle torch
            if 'torch' in module:
                return torch_storages.get(name, TorchTensor)
            # fallback to default
            return pickle.Unpickler.find_class(self, module, name)
        def persistent_load(self, pid):
            return pid

    ret = TorchUnpickler(fb0).load()
    # create key_lookup
    key_lookup = pickle.load(fb0)
    key_real = [None] * len(key_lookup)
    for k,v in key_prelookup.items():
        key_real[key_lookup.index(k)] = v
    # read in the actual data
    for storage_type, obj_size, np_array, np_shape, np_strides in key_real:
        ll = struct.unpack("Q", fb0.read(8))[0]
        assert ll == obj_size
        bytes_size = {np.float32: 4, np.int64: 8}[storage_type]
        mydat = fb0.read(ll * bytes_size)
        np_array[:] = np.frombuffer(mydat, storage_type)
        np_array.shape = np_shape
        real_strides = tuple([x*bytes_size for x in np_strides])
        np_array.strides = real_strides
    return ret
