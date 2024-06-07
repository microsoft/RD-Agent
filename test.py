import pickle
from rdagent.factor_implementation.CoSTEER import CoSTEERFG
from rdagent.factor_implementation.share_modules.factor_implementation_utils import load_data_from_dict
factor_knowledge_base = pickle.load(open("/home/finco/v-wenjunfeng/RD-Agent/factor_dict_original.pkl", "rb"))

factor_tasks = load_data_from_dict(factor_knowledge_base)

factor_generate_method = CoSTEERFG()

result = factor_generate_method.generate(factor_tasks)
