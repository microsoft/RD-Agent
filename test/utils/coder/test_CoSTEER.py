import unittest


class CoSTEERTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def to_str(self, obj):
        return "".join(str(obj).split())

    def test_data_loader(self):
        # 1) Build the data loader task/experiment
        # 2) build an according CoSTEER
        # 3) test the results
        # - check spec.md
        # - check data_loader.py
        pass

    def test_model(self):
        # 1) Build the model experiment/task/workspace from tpl_ex
        # 2) build an according CoSTEER
        # 3) test the results
        pass


if __name__ == "__main__":
    unittest.main()
