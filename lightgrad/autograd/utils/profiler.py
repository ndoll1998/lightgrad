from time import time
from functools import wraps
from collections import defaultdict

class Profiler(object):
    _active_profilers = []
    def __init__(self):
        self.__ft, self.__fc = defaultdict(float), defaultdict(int)
        self.__bt, self.__bc = defaultdict(float), defaultdict(int)
    def update(self, name, time_delta, backward=False):
        (self.__bt if backward else self.__ft)[name] += time_delta
        (self.__bc if backward else self.__fc)[name] += 1

    def __enter__(self, *args):
        Profiler._active_profilers.append(self)
        return self
    def __exit__(self, *args):
        Profiler._active_profilers.remove(self)

    def print(self, topn=-1):
        names = set.union(set(self.__ft.keys()), set(self.__bt.keys()))
        names = sorted(names, key=lambda n: -self.__ft[n])
        names = names[:topn] if topn > 0 else names
        print(" Function       |   forward      \t|   backward   \n" + "-"*70)
        for n in names:
            print(" %-15s| %8.4fs (%i)\t| %8.4fs (%i) " % (
                n, self.__ft[n], self.__fc[n], self.__bt[n], self.__bc[n]))
        print("\n")

class Tracker(object):
    __active = False
    def __init__(self, name:str, backward:bool =False):
        self.update = (lambda td: [p.update(name, td, backward) for p in Profiler._active_profilers]) if not Tracker.__active else lambda td: None
    def __enter__(self, *args):
        Tracker.__active = True
        self.st = time()
    def __exit__(self, *args):
        Tracker.__active = False
        self.update(time() - self.st)
