# TODO: fix the train.py

import importlib.util
from pathlib import Path


def import_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DIRNAME = Path(__file__).absolute().resolve().parent

y = target
X = text[: len(train)]
X_test = text[len(train) :]

for f in DIRNAME.glob("feature/feat*.py"):
    cls = import_module_from_path(f.stem, f).feature_engineering_cls()
    cls.fit(X_train)
    X_train_f = cls.transform(X_train)
    X_test_f = cls.transform(X_test)

    X_train_l.append(X_train_f)
    X_test_l.append(X_test_f)


submission["cohesion"] = predictions[:, 0]
submission["syntax"] = predictions[:, 1]
submission["vocabulary"] = predictions[:, 2]
submission["phraseology"] = predictions[:, 3]
submission["grammar"] = predictions[:, 4]
submission["conventions"] = predictions[:, 5]

submission.to_csv("submission.csv", index=False)  # writing data to a CSV file
