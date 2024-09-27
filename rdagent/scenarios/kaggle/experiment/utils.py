from pathlib import Path

import nbformat as nbf


def python_files_to_notebook(competition: str, py_dir: str):
    py_dir: Path = Path(py_dir)
    save_path: Path = py_dir / "merged.ipynb"

    pre_file = py_dir / "fea_share_preprocess.py"
    pre_py = pre_file.read_text()

    pre_py = pre_py.replace("/kaggle/input", f"/kaggle/input/{competition}")

    fea_files = list(py_dir.glob("feature/*.py"))
    fea_pys = {
        f"{fea_file.stem}_cls": fea_file.read_text().replace("feature_engineering_cls", f"{fea_file.stem}_cls").strip()
        + "()\n"
        for fea_file in fea_files
    }

    model_files = list(py_dir.glob("model/model*.py"))
    model_pys = {f"{model_file.stem}": model_file.read_text().strip() for model_file in model_files}
    for k, v in model_pys.items():
        model_pys[k] = v.replace("def fit(", "def fit(self, ").replace("def predict(", "def predict(self, ")

        lines = model_pys[k].split("\n")
        indent = False
        first_line = -1
        for i, line in enumerate(lines):
            if "def " in line:
                indent = True
                if first_line == -1:
                    first_line = i
            if indent:
                lines[i] = "    " + line
        lines.insert(first_line, f"class {k}:\n")
        model_pys[k] = "\n".join(lines)

    select_files = list(py_dir.glob("model/select*.py"))
    select_pys = {
        f"{select_file.stem}": select_file.read_text().replace("def select(", f"def {select_file.stem}(")
        for select_file in select_files
    }

    train_file = py_dir / "train.py"
    train_py = train_file.read_text()

    train_py = train_py.replace("from fea_share_preprocess import preprocess_script", "")
    train_py = train_py.replace("DIRNAME = Path(__file__).absolute().resolve().parent", "")

    fea_cls_list_str = "[" + ", ".join(list(fea_pys.keys())) + "]"
    train_py = train_py.replace(
        'for f in DIRNAME.glob("feature/feat*.py"):', f"for cls in {fea_cls_list_str}:"
    ).replace("cls = import_module_from_path(f.stem, f).feature_engineering_cls()", "")

    model_cls_list_str = "[" + ", ".join(list(model_pys.keys())) + "]"
    train_py = (
        train_py.replace('for f in DIRNAME.glob("model/model*.py"):', f"for mc in {model_cls_list_str}:")
        .replace("m = import_module_from_path(f.stem, f)", "m = mc()")
        .replace('select_python_path = f.with_name(f.stem.replace("model", "select") + f.suffix)', "")
        .replace(
            "select_m = import_module_from_path(select_python_path.stem, select_python_path)",
            'select_m = eval(mc.__name__.replace("model", "select"))',
        )
        .replace("select_m.select", "select_m")
        .replace("[2].select", "[2]")
    )

    nb = nbf.v4.new_notebook()
    all_py = ""

    nb.cells.append(nbf.v4.new_code_cell(pre_py))
    all_py += pre_py + "\n\n"

    for v in fea_pys.values():
        nb.cells.append(nbf.v4.new_code_cell(v))
        all_py += v + "\n\n"

    for v in model_pys.values():
        nb.cells.append(nbf.v4.new_code_cell(v))
        all_py += v + "\n\n"

    for v in select_pys.values():
        nb.cells.append(nbf.v4.new_code_cell(v))
        all_py += v + "\n\n"

    nb.cells.append(nbf.v4.new_code_cell(train_py))
    all_py += train_py + "\n"

    with save_path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)

    with save_path.with_suffix(".py").open("w", encoding="utf-8") as f:
        f.write(all_py)
