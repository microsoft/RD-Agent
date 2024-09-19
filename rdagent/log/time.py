import time
from functools import wraps


def measure_time(method):
    @wraps(method)
    def timed(*args, **kwargs):
        self = args[0]
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        method_name = method.__name__
        self.times[method_name] = duration
        return result

    return timed
