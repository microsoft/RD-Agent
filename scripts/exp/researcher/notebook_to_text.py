import nbformat
from nbconvert import MarkdownExporter
from pathlib import Path

def convert_notebooks_to_markdown(notebook_root_dir):
    # notebook_root_dir = Path("/data/userdata/share/kaggle/notebooks")

    for competition_folder in notebook_root_dir.iterdir():
        if competition_folder.is_dir():
            for subfolder in competition_folder.iterdir():
                if subfolder.is_dir():
                    ipynb_file = next(subfolder.glob("*.ipynb"), None)
                    if ipynb_file:
                        try:
                            with open(ipynb_file, 'r', encoding='utf-8') as f:
                                notebook = nbformat.read(f, as_version=4)

                            markdown_exporter = MarkdownExporter()
                            (body, resources) = markdown_exporter.from_notebook_node(notebook)

                            markdown_path = subfolder / f"{ipynb_file.stem}.md"
                            with open(markdown_path, 'w', encoding='utf-8') as md_file:
                                md_file.write(body)

                            print(f"Converted {ipynb_file} to Markdown in {markdown_path}")
                        except Exception as e:
                            print(f"Error converting {ipynb_file}: {str(e)}")
                    else:
                        print(f"No .ipynb file found in {subfolder}")

    print("Notebook conversion complete.")


notebook_root_dir = Path("/data/userdata/share/kaggle/notebooks")
convert_notebooks_to_markdown(notebook_root_dir)