"""
我在尝试重构这段代码，减少冗余，并添加新的功能和逻辑。目前的代码里，我会先: step-1. 生成 component, step-2: 再根据 component 生成 idea （ hypothesis 和 task） . 在 step 2 阶段 我们可以选用 BO mode. 生成多个idea 并进行评估，选择分数最高的那个。新的代码和功能应该是包括：

1. step 1 和 step 2 应该被独立封装进 不同的 class method (类似现在的 idea_propose) 

2. 我们 enable 扩展 BO 模式，他允许 将 “batch 生成 - 评估” 这种思路，扩展到 “ component level”。 即我们允许 抽样不同的 component， 每个 component 再抽样不同的 idea, 然后将所有的 candidate 进行评估

3. 尽量减少重复代码 （现在的code中有一些重复代码）

为达成这些目的，你能在新的代码设计上给我一些建议吗

"""



// ... existing code ...



def gen(self, trace: DSTrace, BO_mode: bool = True, BO_step: int = 5) -> DSExperiment:
    # Get configuration settings
    idea_bo_mode = DS_RD_SETTING.idea_bo_mode
    component_bo_mode = DS_RD_SETTING.component_bo_mode
    batch_bo_eval = DS_RD_SETTING.batch_bo_eval
    idea_bo_step = DS_RD_SETTING.idea_bo_step
    component_bo_step = DS_RD_SETTING.component_bo_step

    scenario_desc = trace.scen.get_scenario_all_desc()
    last_successful_exp = trace.last_successful_exp()

    next_missing_component = trace.next_incomplete_component()

    init_component_config = {
        "DataLoadSpec": {"task_cls": DataLoaderTask, "spec_file": None, "component_prompt_key": "data_loader"},
        "FeatureEng": {"task_cls": FeatureTask, "spec_file": "spec/feature.md", "component_prompt_key": "feature"},
        "Model": {"task_cls": ModelTask, "spec_file": "spec/model.md", "component_prompt_key": "model"},
        "Ensemble": {"task_cls": EnsembleTask, "spec_file": "spec/ensemble.md", "component_prompt_key": "ensemble"},
        "Workflow": {"task_cls": WorkflowTask, "spec_file": "spec/workflow.md", "component_prompt_key": "workflow"},
    }

    # If we need to generate a missing component
    if next_missing_component in init_component_config:
        config = init_component_config[next_missing_component]
        return self._handle_missing_component(
            component=next_missing_component,
            task_cls=config["task_cls"],
            scenario_desc=scenario_desc,
            last_successful_exp=last_successful_exp,
            spec_file=config.get("spec_file"),
            trace=trace,
            component_prompt_key=config.get("component_prompt_key"),
        )
    else:  # We need to polish existing components
        sota_exp = trace.sota_experiment()
        assert sota_exp is not None, "SOTA experiment is not provided."
        
        # Prepare context for generation
        context = self._prepare_generation_context(trace, scenario_desc)
        
        # Multi-level Bayesian Optimization
        if component_bo_mode and idea_bo_mode:
            return self._multi_level_bo_generation(
                trace, context, component_bo_step, idea_bo_step, sota_exp
            )
        # Component-level BO only
        elif component_bo_mode:
            return self._component_level_bo_generation(
                trace, context, component_bo_step, idea_bo_step, sota_exp
            )
        # Idea-level BO only (original implementation)
        elif idea_bo_mode:
            component = self._component_propose(trace, context)
            component_info = COMPONENT_TASK_MAPPING.get(component)
            if not component_info:
                raise ValueError(f"Unknown component: {component}")
                
            # Generate multiple ideas and select best
            return self._idea_level_bo_generation(
                trace, context, component, component_info, idea_bo_step, sota_exp
            )
        # Regular mode - single component, single idea
        else:
            component = self._component_propose(trace, context)
            component_info = COMPONENT_TASK_MAPPING.get(component)
            if not component_info:
                raise ValueError(f"Unknown component: {component}")
                
            # Generate single idea
            hypothesis, task, new_workflow_desc = self.idea_propose(
                trace, component, component_info, scenario_desc
            )
            
            # Create experiment
            exp = DSExperiment(pending_tasks_list=[[task]], hypothesis=hypothesis)
            exp.experiment_workspace.inject_code_from_folder(sota_exp.experiment_workspace)
            
            if new_workflow_desc != "No update needed":
                workflow_task = WorkflowTask(
                    name="Workflow",
                    description=new_workflow_desc,
                )
                exp.pending_tasks_list.append([workflow_task])
            return exp

def _prepare_generation_context(self, trace: DSTrace, scenario_desc: str) -> dict:
    """Prepare the common context needed for generation."""
    sota_exp = trace.sota_experiment()
    assert sota_exp is not None, "SOTA experiment is not provided."
    exp_and_feedback = trace.hist[-1]
    last_exp = exp_and_feedback[0]

    # Describe current best solution
    sota_exp_desc = T("scenarios.data_science.share:describe.exp").r(
        exp=sota_exp, heading="Best of previous exploration of the scenario"
    )
    last_exp_diff = "\n".join(
        generate_diff_from_dict(
            sota_exp.experiment_workspace.file_dict, last_exp.experiment_workspace.file_dict
        )
    )

    # Get experiment feedback lists
    sota_exp_feedback_list = trace.experiment_and_feedback_list_after_init(return_type="sota")
    failed_exp_feedback_list = trace.experiment_and_feedback_list_after_init(return_type="failed")[
        -self.max_trace_hist:
    ]
    all_exp_feedback_list = trace.experiment_and_feedback_list_after_init(return_type="all")
    
    # Create dataframe of past experiments
    trace_component_to_feedback_df = pd.DataFrame(columns=["component", "hypothesis", "decision"])
    for index, (exp, fb) in enumerate(all_exp_feedback_list):
        trace_component_to_feedback_df.loc[f"trial {index + 1}"] = [
            exp.hypothesis.component,
            exp.hypothesis.hypothesis,
            fb.decision,
        ]

    # Generate feedback descriptions
    sota_exp_feedback_list_desc = T("scenarios.data_science.share:describe.trace").r(
        exp_and_feedback_list=sota_exp_feedback_list,
        success=True,
    )
    failed_exp_feedback_list_desc = T("scenarios.data_science.share:describe.trace").r(
        exp_and_feedback_list=failed_exp_feedback_list,
        success=False,
    )
    
    return {
        "scenario_desc": scenario_desc,
        "sota_exp": sota_exp,
        "last_exp": last_exp,
        "sota_exp_desc": sota_exp_desc,
        "last_exp_diff": last_exp_diff,
        "sota_exp_feedback_list": sota_exp_feedback_list,
        "failed_exp_feedback_list": failed_exp_feedback_list,
        "all_exp_feedback_list": all_exp_feedback_list,
        "trace_component_to_feedback_df": trace_component_to_feedback_df,
        "sota_exp_feedback_list_desc": sota_exp_feedback_list_desc,
        "failed_exp_feedback_list_desc": failed_exp_feedback_list_desc,
    }

def _component_propose(self, trace: DSTrace, context: dict) -> str:
    """Generate a component proposal."""
    # Generate component using template with proper context
    component_sys_prompt = T(".prompts:component_gen.system").r(
        scenario=context["scenario_desc"],
        sota_exp_desc=context["sota_exp_desc"],
        last_exp_diff=context["last_exp_diff"],
        component_output_format=T(".prompts:output_format.component").r(),
    )

    component_user_prompt = T(".prompts:component_gen.user").r(
        sota_exp_and_feedback_list_desc=context["sota_exp_feedback_list_desc"],
        failed_exp_and_feedback_list_desc=context["failed_exp_feedback_list_desc"],
        component_and_feedback_df=(
            context["trace_component_to_feedback_df"].to_string()
            if len(context["trace_component_to_feedback_df"]) > 0
            else "No experiment and feedback provided"
        ),
    )

    resp_dict_component: dict = json.loads(
        APIBackend().build_messages_and_create_chat_completion(
            component_user_prompt, component_sys_prompt, json_mode=True, json_target_type=Dict[str, str]
        )
    )

    component = resp_dict_component.get("component", "Component not provided")
    
    # Apply heuristic rule
    sota_exp_model_file_count = len(
        [
            k
            for k in context["sota_exp"].experiment_workspace.file_dict.keys()
            if k.endswith(".py") and "test" not in k and k.startswith("model")
        ]
    )
    if sota_exp_model_file_count <= 1 and component == "Ensemble":
        component = "Model"
        
    return component

def _multi_level_bo_generation(self, trace: DSTrace, context: dict, 
                              component_bo_step: int, idea_bo_step: int, 
                              sota_exp: DSExperiment) -> DSExperiment:
    """Generate multiple components and multiple ideas per component, evaluate all, and select best."""
    all_candidates = []
    all_scores = []
    
    # Generate multiple components
    for i in range(component_bo_step):
        component = self._component_propose(trace, context)
        component_info = COMPONENT_TASK_MAPPING.get(component)
        
        if not component_info:
            continue
            
        # For each component, generate multiple ideas
        for j in range(idea_bo_step):
            hypothesis, task, new_workflow_desc = self.idea_propose(
                trace, component, component_info, context["scenario_desc"]
            )
            analysis, est_score = self.idea_evaluate(
                trace, hypothesis, task, component, component_info, context["scenario_desc"]
            )
            
            all_candidates.append((component, hypothesis, task, new_workflow_desc))
            all_scores.append(est_score)
    
    # Select the best candidate
    if not all_candidates:
        # Fallback to regular generation if no candidates were generated
        component = self._component_propose(trace, context)
        component_info = COMPONENT_TASK_MAPPING.get(component)
        hypothesis, task, new_workflow_desc = self.idea_propose(
            trace, component, component_info, context["scenario_desc"]
        )
    else:
        # TODO: Select based on evaluation_metric_direction
        best_idx = all_scores.index(min(all_scores))
        component, hypothesis, task, new_workflow_desc = all_candidates[best_idx]
    
    # Create experiment
    exp = DSExperiment(pending_tasks_list=[[task]], hypothesis=hypothesis)
    exp.experiment_workspace.inject_code_from_folder(sota_exp.experiment_workspace)
    
    if new_workflow_desc != "No update needed":
        workflow_task = WorkflowTask(
            name="Workflow",
            description=new_workflow_desc,
        )
        exp.pending_tasks_list.append([workflow_task])
    return exp

def _component_level_bo_generation(self, trace: DSTrace, context: dict, 
                                  component_bo_step: int, idea_bo_step: int, 
                                  sota_exp: DSExperiment) -> DSExperiment:
    """Generate multiple components, one idea per component, evaluate all, and select best."""
    component_candidates = []
    component_scores = []
    
    # Generate multiple components
    for i in range(component_bo_step):
        component = self._component_propose(trace, context)
        component_info = COMPONENT_TASK_MAPPING.get(component)
        
        if not component_info:
            continue
            
        # Generate a single idea for this component
        hypothesis, task, new_workflow_desc = self.idea_propose(
            trace, component, component_info, context["scenario_desc"]
        )
        analysis, est_score = self.idea_evaluate(
            trace, hypothesis, task, component, component_info, context["scenario_desc"]
        )
        
        component_candidates.append((component, hypothesis, task, new_workflow_desc))
        component_scores.append(est_score)
    
    # Select the best component-idea pair
    if not component_candidates:
        # Fallback to regular generation if no candidates were generated
        component = self._component_propose(trace, context)
        component_info = COMPONENT_TASK_MAPPING.get(component)
        hypothesis, task, new_workflow_desc = self.idea_propose(
            trace, component, component_info, context["scenario_desc"]
        )
    else:
        # TODO: Select based on evaluation_metric_direction
        best_idx = component_scores.index(min(component_scores))
        component, hypothesis, task, new_workflow_desc = component_candidates[best_idx]
    
    # Create experiment
    exp = DSExperiment(pending_tasks_list=[[task]], hypothesis=hypothesis)
    exp.experiment_workspace.inject_code_from_folder(sota_exp.experiment_workspace)
    
    if new_workflow_desc != "No update needed":
        workflow_task = WorkflowTask(
            name="Workflow",
            description=new_workflow_desc,
        )
        exp.pending_tasks_list.append([workflow_task])
    return exp

def _idea_level_bo_generation(self, trace: DSTrace, context: dict, 
                             component: str, component_info: dict, 
                             idea_bo_step: int, sota_exp: DSExperiment) -> DSExperiment:
    """Generate multiple ideas for a single component, evaluate all, and select best."""
    score_list = []
    candidate_list = []
    
    # Generate multiple ideas for the component
    for i in range(idea_bo_step):
        hypothesis, task, new_workflow_desc = self.idea_propose(
            trace, component, component_info, context["scenario_desc"]
        )
        analysis, est_score = self.idea_evaluate(
            trace, hypothesis, task, component, component_info, context["scenario_desc"]
        )
        score_list.append(est_score)
        candidate_list.append((hypothesis, task, new_workflow_desc))

    # Select the best idea
    if not candidate_list:
        # Fallback to regular generation if no candidates were generated
        hypothesis, task, new_workflow_desc = self.idea_propose(
            trace, component, component_info, context["scenario_desc"]
        )
    else:
        # TODO: Select based on evaluation_metric_direction
        best_candidate = candidate_list[score_list.index(min(score_list))]
        hypothesis, task, new_workflow_desc = best_candidate
    
    # Create experiment
    exp = DSExperiment(pending_tasks_list=[[task]], hypothesis=hypothesis)
    exp.experiment_workspace.inject_code_from_folder(sota_exp.experiment_workspace)
    
    if new_workflow_desc != "No update needed":
        workflow_task = WorkflowTask(
            name="Workflow",
            description=new_workflow_desc,
        )
        exp.pending_tasks_list.append([workflow_task])
    return exp

// ... existing code ...