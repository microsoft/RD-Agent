import unittest

from rdagent.core.utils import SingletonBaseClass


class A(SingletonBaseClass):
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class MiscTest(unittest.TestCase):
    def test_singleton(self):
        a1 = A()
        a2 = A()
        a3 = A(x=3)
        a4 = A(x=2)
        a5 = A(b=3)
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


if __name__ == "__main__":
    unittest.main()
