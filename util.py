import timeit
import os
import importlib
import signal

timers = {}
def time_function(func):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        try:
            return func(*args, **kwargs)
        finally:
            runtime = timeit.default_timer() - start
            key = func.__name__
            try:
                timers[key] += runtime
            except KeyError:
                timers[key] = runtime
    return wrapper

def copy_buffers(bufs):
    """Helper function to dealing with lmdb buffers."""
    for buf in bufs:
        yield buf[:]

def open_path(path, mode):
    if mode[0] in ("a", "w"):
        directory = os.path.dirname(path)
        if directory != "" and not os.path.exists(directory):
            os.makedirs(directory)
    return open(path, mode)

class DelayedInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handler)

    def handler(self, *args):
        if self.signal_received:
            self.send_signal()
        print "Signal caught, interrupt again to override..."
        self.signal_received = True
        self.signal_context = args

    def send_signal(self):
        self.old_handler(*self.signal_context)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.send_signal()

