import unittest

import pytest

from rdagent.core.utils import SingletonBaseClass


class A(SingletonBaseClass):
    def __init__(self, **kwargs):
        print(self, "__init__", kwargs)  # make sure the __init__ is called only once.
        self.kwargs = kwargs

    def __str__(self) -> str:
        return f"{self.__class__.__name__}.{getattr(self, 'kwargs', None)}"

    def __repr__(self) -> str:
        return self.__str__()


@pytest.mark.offline
class MiscTest(unittest.TestCase):
    def test_singleton(self):
        print("a1=================")
        a1 = A()
        print("a2=================")
        a2 = A()
        print("a3=================")
        a3 = A(x=3)
        print("a4=================")
        a4 = A(x=2)
        print("a5=================")
        a5 = A(b=3)
        print("a6=================")
        a6 = A(x=3)

        # Check that a1 and a2 are the same instance
        self.assertIs(a1, a2)

        # Check that a3 and a6 are the same instance
        self.assertIs(a3, a6)

        # Check that a1 and a3 are different instances
        self.assertIsNot(a1, a3)

        # Check that a3 and a4 are different instances
        self.assertIsNot(a3, a4)

        # Check that a4 and a5 are different instances
        self.assertIsNot(a4, a5)

        # Check that a5 and a6 are different instances
        self.assertIsNot(a5, a6)

        print(id(a1), id(a2), id(a3), id(a4), id(a5), id(a6))

        print("...................... Start testing pickle ......................")

        # Test pickle
        import pickle

        with self.assertRaises(pickle.PicklingError):
            with open("a3.pkl", "wb") as f:
                pickle.dump(a3, f)
        # NOTE: If the pickle feature is not disabled,
        # loading a3.pkl will return a1, and a1 will be updated with a3's attributes.
        # print(a1.kwargs)
        # with open("a3.pkl", "rb") as f:
        #     a3_pkl = pickle.load(f)
        # print(id(a3), id(a3_pkl))  # not the same object
        # print(a1.kwargs)  # a1 will be changed.


if __name__ == "__main__":
    unittest.main()
