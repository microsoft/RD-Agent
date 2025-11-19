import unittest
from unittest.mock import Mock, patch
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rdagent.components.coder.factor_coder.evolving_strategy import FactorMultiProcessEvolvingStrategy
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.core.experiment import FBWorkspace

@pytest.fixture(autouse=True)
def mock_strategy_init():
    with patch('rdagent.components.coder.factor_coder.evolving_strategy.MultiProcessEvolvingStrategy.__init__', return_value=None):
        yield

@pytest.fixture(autouse=True)
def mock_workspace():
    with patch('rdagent.components.coder.factor_coder.factor.FactorFBWorkspace') as mock_workspace_class:
        mock_workspace_instance = Mock(spec=FBWorkspace)
        mock_workspace_class.return_value = mock_workspace_instance
        yield mock_workspace_class

class TestAssignCodeListToEvoFix(unittest.TestCase):
    def setUp(self):
        self.mock_task = Mock(spec=FactorTask)
        self.mock_task.get_task_information.return_value = "Mock task info"
        self.strategy = FactorMultiProcessEvolvingStrategy(Mock(), Mock())
        self.strategy.scen = Mock()
        self.strategy.scen.get_scenario_all_desc.return_value = "Mock scenario"

    @patch('rdagent.components.coder.factor_coder.evolving_strategy.FactorFBWorkspace.inject_files')
    def test_happy_path_strings_only(self, mock_inject_files):
        code_list = ['def factor1(): return 1 + 2', None, 'print("Another factor")']
        evo = Mock(sub_tasks=[self.mock_task] * 3, sub_workspace_list=[None] * 3)
        result = self.strategy.assign_code_list_to_evo(code_list, evo)
        self.assertIs(evo, result)
        self.assertEqual(mock_inject_files.call_count, 2)  # Only non-None items

    @patch('rdagent.components.coder.factor_coder.evolving_strategy.FactorFBWorkspace.inject_files')
    def test_dict_with_factor_py_key(self, mock_inject_files):
        code_list = [{'factor.py': 'def alpha(): return close / open - 1'}, {'factor.py': ''}, 'fallback str code', {'no_factor.py': 'invalid'}]
        evo = Mock(sub_tasks=[self.mock_task] * 4, sub_workspace_list=[None] * 4)
        result = self.strategy.assign_code_list_to_evo(code_list, evo)
        self.assertEqual(code_list[0], 'def alpha(): return close / open - 1')
        self.assertEqual(mock_inject_files.call_count, 2)  # Only indices 0 and 2 injected

    @patch('rdagent.components.coder.factor_coder.evolving_strategy.FactorFBWorkspace.inject_files')
    def test_dict_without_factor_py_key(self, mock_inject_files):
        code_list = [{'code': 'def momentum(): return vwap / ref(1)'}, {'output': 'model code here'}, None]
        evo = Mock(sub_tasks=[self.mock_task] * 3, sub_workspace_list=[None] * 3)
        result = self.strategy.assign_code_list_to_evo(code_list, evo)
        self.assertEqual(code_list[0], 'def momentum(): return vwap / ref(1)')
        self.assertEqual(mock_inject_files.call_count, 2)  # Indices 0 and 1 injected

    @patch('rdagent.components.coder.factor_coder.evolving_strategy.FactorFBWorkspace.inject_files')
    def test_empty_or_invalid_list(self, mock_inject_files):
        code_list = []
        evo = Mock(sub_tasks=[], sub_workspace_list=[])
        result = self.strategy.assign_code_list_to_evo(code_list, evo)
        self.assertIs(evo, result)
        mock_inject_files.assert_not_called()
        
        code_list = [None, None]
        evo = Mock(sub_tasks=[self.mock_task] * 2, sub_workspace_list=[None] * 2)
        result = self.strategy.assign_code_list_to_evo(code_list, evo)
        mock_inject_files.assert_not_called()

    def test_unfixed_behavior_simulation(self):
        def broken_assign(self, code_list, evo):
            for index in range(len(evo.sub_tasks)):
                if code_list[index] is None:
                    continue
                if evo.sub_workspace_list[index] is None:
                    evo.sub_workspace_list[index] = FactorFBWorkspace(target_task=evo.sub_tasks[index])
                evo.sub_workspace_list[index].inject_files(**{"factor.py": code_list[index]})
            return evo
        
        original_method = self.strategy.assign_code_list_to_evo
        self.strategy.assign_code_list_to_evo = broken_assign.__get__(self.strategy, FactorMultiProcessEvolvingStrategy)
        
        code_list = [{'code': 'bad dict'}]
        evo = Mock(sub_tasks=[self.mock_task], sub_workspace_list=[None])
        
        with self.assertRaises(TypeError):
            self.strategy.assign_code_list_to_evo(code_list, evo)
        
        self.strategy.assign_code_list_to_evo = original_method

if __name__ == '__main__':
    unittest.main(verbosity=2)