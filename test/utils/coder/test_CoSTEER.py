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
        from rdagent.components.coder.data_science.raw_data_loader.test import (
            develop_one_competition,
        )

        exp = develop_one_competition("aerial-cactus-identification")

        pass

    def test_model(self):
        # 1) Build the model experiment/task/workspace from tpl_ex
        # 2) build an according CoSTEER
        # 3) test the results
        from rdagent.components.coder.data_science.model.test import (
            develop_one_competition,
        )

        exp = develop_one_competition("aerial-cactus-identification")

        pass


if __name__ == "__main__":
    unittest.main()
    # pytest test/utils/coder/test_CoSTEER.py
