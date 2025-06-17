class DSProposalV2ExpGen(ExpGen):
    def identify_scenario_problem(self, scenario_desc: str, sota_exp_desc: str) -> Dict:
        sys_prompt = T(".prompts_v2:scenario_problem.system").r(
            problem_spec=T(".prompts_v2:specification.problem").r(),
            problem_output_format=T(".prompts_v2:output_format.problem").r(),
        )
        user_prompt = T(".prompts_v2:scenario_problem.user").r(
            scenario_desc=scenario_desc,
            sota_exp_desc=sota_exp_desc,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, Dict[str, str]],
        )
        return json.loads(response)

    def identify_feedback_problem(
        self, scenario_desc: str, exp_feedback_list_desc: str, sota_exp_desc: str, inject_diverse: bool = False
    ) -> Dict:
        sys_prompt = T(".prompts_v2:feedback_problem.system").r(
            problem_spec=T(".prompts_v2:specification.problem").r(),
            problem_output_format=T(".prompts_v2:output_format.problem").r(),
            inject_diverse=inject_diverse,
        )
        user_prompt = T(".prompts_v2:feedback_problem.user").r(
            scenario_desc=scenario_desc,
            exp_and_feedback_list_desc=exp_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, Dict[str, str]],
        )
        return json.loads(response)

    def identify_problem(
        self, current_sub_trace, scenario_desc, sota_exp_desc, exp_feedback_list_desc, inject_diverse
    ) -> Dict:
        sota_exp_num = sum(1 for _, fb in current_sub_trace if fb.decision)
        failed_exp_num = len(current_sub_trace) - sota_exp_num
        weighted_exp_num = (sota_exp_num * 3 + failed_exp_num * 2) // 2
        self.scen_prob_multiplier = max(0, 3 - weighted_exp_num // 4)

        all_problems = {}
        if self.scen_prob_multiplier > 0:
            scen_problems = self.identify_scenario_problem(
                scenario_desc=scenario_desc,
                sota_exp_desc=sota_exp_desc,
            )
            for problem_name in scen_problems:
                scen_problems[problem_name]["label"] = "SCENARIO_PROBLEM"
                all_problems[problem_name] = scen_problems[problem_name]

        if self.scen_prob_multiplier < 3:
            fb_problems = self.identify_feedback_problem(
                scenario_desc=scenario_desc,
                exp_feedback_list_desc=exp_feedback_list_desc,
                sota_exp_desc=sota_exp_desc,
                inject_diverse=inject_diverse,
            )
            for problem_name in fb_problems:
                fb_problems[problem_name]["label"] = "FEEDBACK_PROBLEM"
                all_problems[problem_name] = fb_problems[problem_name]
        return all_problems

    @wait_retry(retry_n=5)
    def hypothesis_gen(
        self,
        component_desc: str,
        scenario_desc: str,
        exp_feedback_list_desc: str,
        sota_exp_desc: str,
        problems: dict,
        pipeline: bool,
        enable_idea_pool: bool,
        inject_diverse: bool = False,
    ) -> Dict:
        problem_formatted_str = ""
        for problem_name, problem_dict in problems.items():
            if "problem" not in problem_dict:
                continue
            problem_formatted_str += f"Problem Name: {problem_name}\n"
            problem_formatted_str += f"- Problem Description: {problem_dict['problem']}\n"
            if "idea" in problem_dict:
                idea_formatted_str = DSIdea(problem_dict["idea"]).to_formatted_str()
                problem_formatted_str += f"- Sampled Idea by user: \n{idea_formatted_str}\n"
            problem_formatted_str += "\n\n"

        sys_prompt = T(".prompts_v2:hypothesis_gen.system").r(
            component_desc=component_desc,
            hypothesis_spec=T(".prompts_v2:specification.hypothesis").r(pipeline=pipeline),
            hypothesis_output_format=T(".prompts_v2:output_format.hypothesis").r(
                pipeline=pipeline, enable_idea_pool=enable_idea_pool
            ),
            pipeline=pipeline,
            enable_idea_pool=enable_idea_pool,
            inject_diverse=inject_diverse,
        )
        user_prompt = T(".prompts_v2:hypothesis_gen.user").r(
            scenario_desc=scenario_desc,
            exp_and_feedback_list_desc=exp_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
            problems=problem_formatted_str,
            enable_idea_pool=enable_idea_pool,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, Dict[str, str | Dict[str, str | int]]],
        )
        resp_dict = json.loads(response)

        # make sure the problem name is aligned
        problem_keys = set(problems.keys())
        resp_keys = set(resp_dict.keys())
        if not resp_keys.issubset(problem_keys):
            logger.error("Problem names are not fully aligned. Retrying...")
            raise ValueError("Problem names are not fully aligned.")

        return resp_dict

    def compute_top_scores(
        self,
        hypothesis_dict: dict,
    ) -> pd.Series:
        """
        Compute weighted total scores for each hypothesis and return the top five.
        """
        weights = {
            "alignment_score": 0.2,
            "impact_score": 0.4,
            "novelty_score": 0.2,
            "feasibility_score": 0.1,
            "risk_reward_balance_score": 0.1,
        }
        scores_dict = {}
        for problem_name in hypothesis_dict:
            if "hypothesis" not in hypothesis_dict[problem_name]:
                continue
            scores_dict[problem_name] = {}
            for score_key in weights:
                if score_key not in hypothesis_dict[problem_name]["evaluation"]:
                    scores_dict[problem_name][score_key] = 0
                else:
                    try:
                        scores_dict[problem_name][score_key] = (
                            float(hypothesis_dict[problem_name]["evaluation"][score_key]) * weights[score_key]
                        )
                    except (ValueError, TypeError):
                        scores_dict[problem_name][score_key] = 0

        scores = pd.DataFrame(scores_dict)
        scores_sorted = scores.sum().sort_values(ascending=False)
        return scores_sorted[:5]

    def select_hypothesis(
        self,
        scores_sorted: pd.Series,
        hypothesis_dict: dict,
        problem_dict: dict,
    ) -> int:
        """
        From the top five hypotheses (by weighted score), select one based on additional weighting rules
        for 'inspired' flag and 'SCENARIO_PROBLEM' label. Returns the chosen hypothesis name and a
        DSHypothesis instance.
        """
        # Increase the weight of the hypothesis that is inspired by the idea pool to 3x.
        # Linear decay the weight of the scenario problem from 3x to 0x.
        index_to_pick_pool_list = []
        for j, problem_name in enumerate(scores_sorted.index):
            if hypothesis_dict[problem_name].get("inspired", False):
                index_to_pick_pool_list.extend([j] * 2)
            if problem_dict[problem_name]["label"] == "SCENARIO_PROBLEM":
                index_to_pick_pool_list.extend([j] * self.scen_prob_multiplier)
            elif problem_dict[problem_name]["label"] == "FEEDBACK_PROBLEM":
                index_to_pick_pool_list.extend([j] * (3 - self.scen_prob_multiplier))
            else:
                index_to_pick_pool_list.extend([j] * 1)
        logger.info(f"index_to_pick_pool_list: {index_to_pick_pool_list}")

        # Create a random but reproducible integer
        reproducible_int = int.from_bytes(bytes.fromhex(md5_hash(scores_sorted.to_string())), byteorder="big") % len(
            index_to_pick_pool_list
        )
        return index_to_pick_pool_list[reproducible_int]

    def hypothesis_rank(
        self, hypothesis_dict: dict, problem_dict: dict, selected_idx: Optional[int] = None
    ) -> Tuple[str, DSHypothesis]:
        """
        Wrapper method that computes the top five hypotheses by weighted scoring and then selects one
        according to additional weighting rules.
        """
        scores_sorted = self.compute_top_scores(hypothesis_dict)
        if selected_idx is None:
            selected_idx = self.select_hypothesis(
                scores_sorted=scores_sorted, hypothesis_dict=hypothesis_dict, problem_dict=problem_dict
            )

        max_score_problem_name = scores_sorted.index[selected_idx]
        problem_dict = problem_dict.get(max_score_problem_name, {})

        return max_score_problem_name, DSHypothesis(
            component=hypothesis_dict[max_score_problem_name].get("component", "Model"),
            hypothesis=hypothesis_dict[max_score_problem_name].get("hypothesis", "Hypothesis not provided"),
            reason=hypothesis_dict[max_score_problem_name].get("reason", "Reason not provided"),
            problem_name=max_score_problem_name,
            problem_desc=problem_dict.get("problem", "Problem description not provided"),
            problem_label=problem_dict.get("label", "FEEDBACK_PROBLEM"),
        )

    def task_gen(
        self,
        component_desc: str,
        scenario_desc: str,
        sota_exp_desc: str,
        sota_exp: DSExperiment,
        hypothesis: DSHypothesis,
        pipeline: bool,
        failed_exp_feedback_list_desc: str,
    ) -> DSExperiment:
        if pipeline:
            component_info = get_component("Pipeline")
        else:
            component_info = get_component(hypothesis.component)
        if pipeline:
            task_spec = T(f"scenarios.data_science.share:component_spec.Pipeline").r()
        elif DS_RD_SETTING.spec_enabled and sota_exp is not None:
            task_spec = sota_exp.experiment_workspace.file_dict[component_info["spec_file"]]
        else:
            task_spec = T(f"scenarios.data_science.share:component_spec.{hypothesis.component}").r()
        sys_prompt = T(".prompts_v2:task_gen.system").r(
            targets=component_info["target_name"],
            task_specification=task_spec,
            task_output_format=component_info["task_output_format"],
            component_desc=component_desc,
            workflow_check=not pipeline and hypothesis.component != "Workflow",
        )
        user_prompt = T(".prompts_v2:task_gen.user").r(
            scenario_desc=scenario_desc,
            sota_exp_desc=sota_exp_desc,
            hypothesis=str(hypothesis),
            failed_exp_and_feedback_list_desc=failed_exp_feedback_list_desc,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | Dict[str, str]],
        )
        task_dict = json.loads(response)
        task_design = task_dict.get("task_design", {})
        task_name = (
            task_design["model_name"] if (hypothesis.component == "Model" and not pipeline) else hypothesis.component
        )
        description = (
            task_design
            if isinstance(task_design, str)
            else task_design.get("description", f"{component_info['target_name']} description not provided")
        )
        task_class = component_info["task_class"]
        task = task_class(
            name=task_name,
            description=description,
        )
        new_workflow_desc = task_dict.get("workflow_update", "No update needed")
        exp = DSExperiment(pending_tasks_list=[[task]], hypothesis=hypothesis)
        # exp.experiment_workspace.inject_code_from_folder(sota_exp.experiment_workspace.workspace_path)
        if sota_exp is not None:
            exp.experiment_workspace.inject_code_from_file_dict(sota_exp.experiment_workspace)
        if not pipeline and new_workflow_desc != "No update needed":
            workflow_task = WorkflowTask(
                name="Workflow",
                description=new_workflow_desc,
            )
            exp.pending_tasks_list.append([workflow_task])
        return exp

    def gen(self, trace: DSTrace) -> DSExperiment:
        pipeline = DS_RD_SETTING.coder_on_whole_pipeline
        if not pipeline and (draft_exp := draft_exp_in_decomposition(self.scen, trace)):
            return draft_exp

        if pipeline:
            component_desc = T("scenarios.data_science.share:component_description_in_pipeline").r()
        else:
            component_desc = "\n".join(
                [
                    f"[{key}] {value}"
                    for key, value in T("scenarios.data_science.share:component_description").template.items()
                ]
            )

        sota_exp = trace.sota_experiment()
        if not isinstance(sota_exp, DSExperiment):
            eda_output = None
        else:
            eda_output = sota_exp.experiment_workspace.file_dict.get("EDA.md", None)
        scenario_desc = trace.scen.get_scenario_all_desc(eda_output=eda_output)

        sota_exp_desc = T("scenarios.data_science.share:describe.exp").r(
            exp=sota_exp, heading="Best of previous exploration of the scenario"
        )

        exp_feedback_list_desc = T("scenarios.data_science.share:describe.trace").r(
            exp_and_feedback_list=trace.experiment_and_feedback_list_after_init(return_type="all"),
            type="all",
            pipeline=pipeline,
        )
        failed_exp_feedback_list_desc = T("scenarios.data_science.share:describe.trace").r(
            exp_and_feedback_list=trace.experiment_and_feedback_list_after_init(return_type="failed"),
            type="failed",
            pipeline=pipeline,
        )

        if DS_RD_SETTING.enable_inject_diverse and len(trace.hist) > 0:
            if len(trace.current_selection) == 0:
                # start a new sub-trace, and inject diverse problems.
                inject_diverse = True
                logger.info("Start a new sub-trace, and inject diverse problems.")
            else:
                inject_diverse = False
        else:
            inject_diverse = False

        # Step 1: Identify problems
        all_problems = self.identify_problem(
            current_sub_trace=trace.get_parent_exps(),
            scenario_desc=scenario_desc,
            sota_exp_desc=sota_exp_desc,
            exp_feedback_list_desc=exp_feedback_list_desc,
            inject_diverse=inject_diverse,
        )

        # Step 1.5: Sample ideas from idea pool
        if DS_RD_SETTING.enable_knowledge_base:
            all_problems = trace.knowledge_base.sample_ideas(
                problems=all_problems,
                scenario_desc=scenario_desc,
                exp_feedback_list_desc=exp_feedback_list_desc,
                sota_exp_desc=sota_exp_desc,
                competition_desc=self.scen.get_competition_full_desc(),
            )

        # Step 2: Propose hypothesis based on the identified problems (and sampled ideas)
        hypothesis_dict = self.hypothesis_gen(
            component_desc=component_desc,
            scenario_desc=scenario_desc,
            exp_feedback_list_desc=exp_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
            problems=all_problems,
            pipeline=pipeline,
            enable_idea_pool=DS_RD_SETTING.enable_knowledge_base,
            inject_diverse=inject_diverse,
        )
        if not pipeline:
            sota_exp_model_file_count = len(
                [
                    k
                    for k in sota_exp.experiment_workspace.file_dict.keys()
                    if k.endswith(".py") and "test" not in k and k.startswith("model")
                ]
            )
            if sota_exp_model_file_count <= 1:
                pop_names = []
                for problem_name in hypothesis_dict:
                    if hypothesis_dict[problem_name].get("component", "") == "Ensemble":
                        pop_names.append(problem_name)
                for name in pop_names:
                    hypothesis_dict.pop(name)

        # Step 3: Select the best hypothesis
        pickled_problem_name, new_hypothesis = self.hypothesis_rank(
            hypothesis_dict=hypothesis_dict,
            problem_dict=all_problems,
        )
        # Step 3.5: Update knowledge base with the picked problem
        if DS_RD_SETTING.enable_knowledge_base:
            trace.knowledge_base.update_pickled_problem(all_problems, pickled_problem_name)

        return self.task_gen(
            component_desc=component_desc,
            scenario_desc=scenario_desc,
            sota_exp_desc=sota_exp_desc,
            sota_exp=sota_exp,
            hypothesis=new_hypothesis,
            pipeline=pipeline,
            failed_exp_feedback_list_desc=failed_exp_feedback_list_desc,
        )
