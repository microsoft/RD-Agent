"""
This is a class that try to store/resume/traceback the workflow session


Postscripts:
- Originally, I want to implement it in a more general way with python generator.
  However, Python generator is not picklable (dill does not support pickle as well)

"""
import pickle


from collections import defaultdict
from dataclasses import dataclass
import datetime
from typing import Callable
from rdagent.log import rdagent_logger as logger


class LoopMeta(type):
    meta_attr = "meta attribute will become the attribute of class"
    # But it will not present in __init__, __new__, __call__

    def __new__(cls, clsname, bases, attrs):
        # MetaClass的new代表创建子类， Class的new代表创建实例
        # cls 就类似于静态方法
        # - 比较奇妙的地方是它虽然是静态方法，但是不需要静态方法装饰器
        print("创建class之前可以做点什么", clsname, bases, attrs)

        print("这里直接给子类加了个方法")
        # attrs["foo"] = foo

        # move custommized steps into steps
        steps = []
        for name in attrs.keys():
            if not name.startswith("__"):
                steps.append(name)
        attrs["steps"] = steps

        return super().__new__(cls, clsname, bases, attrs)

    def __init__(self, clsname, bases, attrs):
        # MetaClass的init代表初始化子类， Class的init代表初始化实例
        print("创建class之后可以做点什么", clsname, bases, attrs)

    def __call__(self, *args, **kwargs):
        # MetaClass的call代表调用创建的子类( 即创建实例), Class的call代表调用创建的实例
        # 比如 E("test") 会调用 <class '__main__.E'> ('test',) {}
        print("在meta class创建出来实例初始化instance时会调用 `__call__`", self, args, kwargs)
        # 到这一行时还没有初始化， 到下面一行才会初始化
        return super().__call__(*args, **kwargs)


@dataclass
class LoopTrace:
    start: datetime.datetime  # the start time of the trace
    end: datetime.datetime  # the end time of the trace
    # TODO: more information about the trace


class LoopBase:
    steps: list[Callable]
    loop_trace: dict[int, list[LoopTrace]]

    def __init__(self):
        self.loop_idx = 0 # current loop index
        self.step_idx = 0 # the index of next step to be run
        self.loop_prev_out = {} # the step results of current loop
        self.loop_trace = defaultdict(list[LoopTrace])  # the key is the number of loop
        self.session_folder = logger.log_trace_path / "__session__"

    def run(self):
        while True:
            li, si = self.loop_idx, self.step_idx

            start = datetime.datetime.now()

            name = self.steps[si]
            func = getattr(self, name)
            self.loop_prev_out[name] = func(self.loop_prev_out)

            end = datetime.datetime.now()

            self.loop_trace[li].append(LoopTrace(start, end))

            # index increase and save session
            self.step_idx = (self.step_idx + 1) % len(self.steps)
            if self.step_idx == 0:  # reduce to step 0 in next round
                self.loop_idx += 1
                self.loop_prev_out = {}

            self.dump_session(self.session_folder / f"{li}" / f"{si}_{name}") # save a snapshot after the session

    def dump_session(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
