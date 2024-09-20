import time
from functools import wraps
from rdagent.log import rdagent_logger as logger

def measure_time(method):
    @wraps(method)
    def timed(*args, **kwargs):
        with logger.tag("time"):
            start_time = time.time()
            result = method(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            method_name = method.__name__
            logger.log(f"{method_name} took {duration:.2f} sec")
            return result
    return timed
