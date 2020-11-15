from time import time
from functools import wraps
from collections import defaultdict

class __ProfilerMeta(type):
    def __enter__(cls, *args):
        cls._active = True
    def __exit__(cls, *args):
        cls._active = False
class Profiler(object, metaclass=__ProfilerMeta):
    # is profiler active
    _active = False
    __ft, __fc = defaultdict(float), defaultdict(int)
    __bt, __bc = defaultdict(float), defaultdict(int)
    class __Tracker(object):
        def __init__(self, name, backward):
            self.update = lambda td: Profiler.update(name, td, backward)
        def __enter__(self, *args):
            self.st = time()
        def __exit__(self, *args):
            self.update(time() - self.st)
    @staticmethod
    def update(name, time_delta, backward=False):
        if Profiler._active:
            (Profiler.__bt if backward else Profiler.__ft)[name] += time_delta
            (Profiler.__bc if backward else Profiler.__fc)[name] += 1
    @staticmethod
    def profile(name, backward=False):
        return Profiler.__Tracker(name, backward=backward)
    @staticmethod
    def print():
        names = set(list(Profiler.__ft.keys()) + list(Profiler.__bt.keys()))
        names = sorted(names, key=lambda n: -Profiler.__ft[n])
        print(" Function       |   forward      \t|   backward   ")
        print("-"*70)
        for n in names:
            print(" %-15s| %8.4f ms (%i)\t| %8.4f ms (%i) " % (
                n, Profiler.__ft[n], Profiler.__fc[n], Profiler.__bt[n], Profiler.__bc[n]))
        print("\n")