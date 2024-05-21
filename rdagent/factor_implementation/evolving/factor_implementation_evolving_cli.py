import json
import pickle
import subprocess
from pathlib import Path

import pandas as pd
from fire.core import Fire
from tqdm import tqdm

from core.evolving_framework import EvoAgent, KnowledgeBase
from factor_implementation.evolving.evaluators import (
    FactorImplementationEvaluatorV1,
    FactorImplementationsMultiEvaluator,
)
from factor_implementation.evolving.evolvable_subjects import (
    FactorImplementationList,
)
from factor_implementation.evolving.evolving_strategy import (
    FactorEvolvingStrategy,
    FactorEvolvingStrategyWithGraph,
)
from factor_implementation.evolving.knowledge_management import (
    FactorImplementationGraphKnowledgeBase,
    FactorImplementationGraphRAGStrategy,
    FactorImplementationKnowledgeBaseV1,
    FactorImplementationRAGStrategyV1,
)
from factor_implementation.share_modules.factor import (
    FactorImplementationTask,
    FileBasedFactorImplementation,
)
from core.utils import multiprocessing_wrapper

ALPHA101_INIT_COMPONENTS = [
    "1. abs(): absolute value to certain columns",
    "2. log(): log value to certain columns",
    "3. sign(): sign value to certain columns",
    "4. add_two_columns(): add two columns",
    "5. minus_two_columns(): minus two columns",
    "6. times_two_columns(): times two columns",
    "7. divide_two_columns(): divide two columns",
    "8. add_value_to_columns(): add value to columns",
    "9. minus_value_to_columns(): minus value to columns",
    "10. rank(): cross-sectional rank value to columns",
    "11. delay(): value of each data d days ago",
    "12. correlation(): time-serial correlation of column_left and column_right for the past d days",
    "13. covariance(): time-serial covariance of column_left and column_right for the past d days",
    "14. scale_to_a(): scale the columns to sum(abs(x)) is a",
    "15. delta(): today’s value of x minus the value of x d days ago",
    "16. signedpower(): x^a",
    "17. decay_linear(): weighted moving average over the past d days with linearly decaying weights d, d – 1, …, 1 (rescaled to sum up to 1)",
    "18. indneutralize(): x cross-sectionally neutralized against groups g (subindustries, industries, sectors, etc.), i.e., x is cross-sectionally demeaned within each group g",
    "19. ts_min(): time-series min over the past d days, operator min applied across the time-series for the past d days; non-integer number of days d is converted to floor(d)",
    "20. ts_max(): time-series max over the past d days, operator max applied across the time-series for the past d days; non-integer number of days d is converted to floor(d)",
    "21. ts_argmax(): which day ts_max(x, d) occurred on",
    "22. ts_argmin(): which day ts_min(x, d) occurred on",
    "23. ts_rank(): time-series rank in the past d days",
    "24. min(): ts_min(x, d)",
    "25. max(): ts_max(x, d)",
    "26. sum(): time-series sum over the past d days",
    "27. product(): time-series product over the past d days",
    "28. stddev(): moving time-series standard deviation over the past d days",
]


class FactorImplementationEvolvingCli:
    # TODO: we should use polymorphism to load knowledge base, strategies instead of evolving_version
    # TODO: Can we refactor FactorImplementationEvolvingCli into a learning framework to differentiate our learning paradiagm with other ones by iteratively retrying?
    def __init__(self, evolving_version=2) -> None:
        self.evolving_version = evolving_version
        self.knowledge_base = None
        self.latest_factor_implementations = None

    def run_evolving_framework(
        self,
        factor_implementations: FactorImplementationList,
        factor_knowledge_base: KnowledgeBase,
        max_loops: int = 20,
        with_knowledge: bool = True,
        with_feedback: bool = True,
        knowledge_self_gen: bool = True,
    ) -> FactorImplementationList:
        """
        Main target: Implement factors.
        The system also leverages the former knowledge to help implement the factors. Also, new knowledge might be generated during the implementation to help the following implementation.
        The gt_code and gt_value in the Factor instance is used to evaluate the implementation, and the feedback is used to generate high-quality knowledge which helps the agent to evolve.
        """
        es = FactorEvolvingStrategyWithGraph() if self.evolving_version == 2 else FactorEvolvingStrategy()
        rag = (
            FactorImplementationGraphRAGStrategy(factor_knowledge_base)
            if self.evolving_version == 2
            else FactorImplementationRAGStrategyV1(factor_knowledge_base)
        )
        factor_evaluator = FactorImplementationsMultiEvaluator(FactorImplementationEvaluatorV1())
        ea = EvoAgent(es, rag=rag)

        for _ in tqdm(range(max_loops), "Implementing factors"):
            factor_implementations = ea.step_evolving(
                factor_implementations,
                factor_evaluator,
                with_knowledge=with_knowledge,
                with_feedback=with_feedback,
                knowledge_self_gen=knowledge_self_gen,
            )
        return factor_implementations

    def load_or_init_knowledge_base(self, former_knowledge_base_path: Path = None, component_init_list: list = []):
        if former_knowledge_base_path is not None and former_knowledge_base_path.exists():
            factor_knowledge_base = pickle.load(open(former_knowledge_base_path, "rb"))
            if self.evolving_version == 1 and not isinstance(
                factor_knowledge_base, FactorImplementationKnowledgeBaseV1
            ):
                raise ValueError("The former knowledge base is not compatible with the current version")
            elif self.evolving_version == 2 and not isinstance(
                factor_knowledge_base,
                FactorImplementationGraphKnowledgeBase,
            ):
                raise ValueError("The former knowledge base is not compatible with the current version")
        else:
            factor_knowledge_base = (
                FactorImplementationGraphKnowledgeBase(
                    init_component_list=component_init_list,
                )
                if self.evolving_version == 2
                else FactorImplementationKnowledgeBaseV1()
            )
        return factor_knowledge_base

    def implement_factors(
        self,
        factor_implementations: FactorImplementationList,
        former_knowledge_base_path: Path = None,
        new_knowledge_base_path: Path = None,
        component_init_list: list = [],
        max_loops: int = 20,
    ):
        factor_knowledge_base = self.load_or_init_knowledge_base(
            former_knowledge_base_path=former_knowledge_base_path,
            component_init_list=component_init_list,
        )

        new_factor_implementations = self.run_evolving_framework(
            factor_implementations=factor_implementations,
            factor_knowledge_base=factor_knowledge_base,
            max_loops=max_loops,
            with_knowledge=True,
            with_feedback=True,
            knowledge_self_gen=True,
        )
        if new_knowledge_base_path is not None:
            pickle.dump(factor_knowledge_base, open(new_knowledge_base_path, "wb"))
        self.knowledge_base = factor_knowledge_base
        self.latest_factor_implementations = factor_implementations
        return new_factor_implementations

    def _read_alpha101_factors(
        self,
        alpha101_evo_subs_path: Path = None,
        alpha101_data_path=Path().cwd() / "git_ignore_folder" / "alpha101_related_files",
        start_index=0,
        end_index=32,
        read_gt_factors=True,
    ) -> FactorImplementationList:
        """
        Read the alpha101 factors from the alpha101_related_files folder
        """
        if alpha101_evo_subs_path is not None and alpha101_evo_subs_path.exists():
            factor_implementations = pickle.load(open(alpha101_evo_subs_path, "rb"))
        else:
            target_factor_plain_list = json.load(
                open(alpha101_data_path / "target_factor_task_list.json"),
            )
            name_to_code = json.load(open(alpha101_data_path / "name_to_code.json"))
            gt_df = pd.read_hdf(alpha101_data_path / "gt_filtered.h5")

            # First read the target factor task
            target_factor_tasks = []
            for factor_list_item in target_factor_plain_list:
                target_factor_tasks.append(
                    FactorImplementationTask(
                        factor_name=factor_list_item[0],
                        factor_description=factor_list_item[1],
                        factor_formulation=factor_list_item[2],
                        factor_formulation_description=factor_list_item[3],
                    ),
                )

            # Second read the gt factor implementations
            corresponding_gt_implementations = []
            for factor_task in target_factor_tasks:
                name = factor_task.factor_name
                gt_code = name_to_code[name]
                gt_value = gt_df.loc(axis=1)[[name]]
                corresponding_gt_implementations.append(
                    FileBasedFactorImplementation(
                        code=gt_code,
                        executed_factor_value_dataframe=gt_value,
                        target_task=factor_task,
                    ),
                )

            # Finally generate the factor implementations as evolvable subjects
            factor_implementations = FactorImplementationList(
                target_factor_tasks=target_factor_tasks,
                corresponding_gt_implementations=(corresponding_gt_implementations if read_gt_factors else None),
            )

        factor_implementations.target_factor_tasks = factor_implementations.target_factor_tasks[start_index:end_index]
        factor_implementations.corresponding_gt_implementations = (
            factor_implementations.corresponding_gt_implementations[start_index:end_index] if read_gt_factors else None
        )

        return factor_implementations

    def implement_alpha101(
        self,
        max_loops=30,
    ) -> FactorImplementationList:
        """
        Implement the alpha101 factors to gather knowledge TODO: implement the code
        """
        factor_implementations = self._read_alpha101_factors(
            alpha101_evo_subs_path=Path.cwd() / "alpha101_evo_subs.pkl",
            start_index=0,
            end_index=64,
            read_gt_factors=True,
        )
        self.implement_factors(
            factor_implementations,
            former_knowledge_base_path=Path.cwd()
            / f"alpha101_knowledge_base_v{self.evolving_version}_project_product.pkl",
            new_knowledge_base_path=Path.cwd()
            / f"alpha101_knowledge_base_v{self.evolving_version}_project_product.pkl",
            component_init_list=ALPHA101_INIT_COMPONENTS,
            max_loops=100,
        )

        factor_implementations = self._read_alpha101_factors(
            alpha101_evo_subs_path=Path.cwd() / "alpha101_evo_subs.pkl",
            start_index=64,
            end_index=96,
            read_gt_factors=False,
        )
        final_imp = self.implement_factors(
            factor_implementations,
            former_knowledge_base_path=Path.cwd()
            / f"alpha101_knowledge_base_v{self.evolving_version}_project_product.pkl",
            new_knowledge_base_path=Path.cwd()
            / f"alpha101_knowledge_base_v{self.evolving_version}_self_evolving_project_product.pkl",
            component_init_list=ALPHA101_INIT_COMPONENTS,
            max_loops=10,
        )
        final_imp.corresponding_gt_implementations = factor_implementations = self._read_alpha101_factors(
            alpha101_evo_subs_path=Path.cwd() / "alpha101_evo_subs.pkl",
            start_index=64,
            end_index=96,
            read_gt_factors=True,
        ).corresponding_gt_implementations

        feedbacks = FactorImplementationsMultiEvaluator().evaluate(final_imp)
        print([feedback.final_decision if feedback is not None else None for feedback in feedbacks].count(True))

    def implement_amc(
        self, evo_sub_path_str, former_knowledge_base_path_str, implementation_dump_path_str, slice_index
    ):
        factor_implementations: FactorImplementationList = pickle.load(open(evo_sub_path_str, "rb"))
        factor_implementations.target_factor_tasks = factor_implementations.target_factor_tasks[
            slice_index * 16 : slice_index * 16 + 16
        ]
        if len(factor_implementations.target_factor_tasks) == 0:
            return
        if Path(implementation_dump_path_str).exists():
            return
        factor_implementations = self.implement_factors(
            factor_implementations,
            former_knowledge_base_path=Path(former_knowledge_base_path_str),
            component_init_list=ALPHA101_INIT_COMPONENTS,
            max_loops=10,
        )
        pickle.dump(factor_implementations, open(implementation_dump_path_str, "wb"))

    def execute_command(self, command, cwd):
        print(command, cwd)
        try:
            subprocess.check_output(
                command,
                shell=True,
                cwd=cwd,
            )
        except subprocess.CalledProcessError as e:
            print(e.output.decode())

    def multi_inference_amc_factors(self, type):
        slice_count = {"price_volume": 35, "fundamental": 24, "high_frequency": 16}[type]
        res = multiprocessing_wrapper(
            [
                (
                    self.execute_command,
                    (
                        f"python src/scripts/factor_implementation/baselines/evolving/factor_implementation_evolving_cli.py implement_amc ./{type}_factors.pkl ./knowledge_base_v2_with_alpha101_and_10_factors.pkl ./inference_amc_factors_{type}_{slice}.pkl {slice}",
                        Path.cwd(),
                    ),
                )
                for slice in range(slice_count)
            ],
            n=2,
        )


if __name__ == "__main__":
    Fire(FactorImplementationEvolvingCli)
