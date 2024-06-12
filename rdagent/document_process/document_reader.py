from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader

if TYPE_CHECKING:
    from langchain_core.documents import Document
from rdagent.core.conf import FincoSettings as Config


def load_documents_by_langchain(path: Path) -> list:
    """Load documents from the specified path.

    Args:
        path (str): The path to the directory or file containing the documents.

    Returns:
        list: A list of loaded documents.
    """
    loader = PyPDFDirectoryLoader(str(path), silent_errors=True) if path.is_dir() else PyPDFLoader(str(path))
    return loader.load()


def process_documents_by_langchain(docs: list[Document]) -> dict[str, str]:
    """Process a list of documents and group them by document name.

    Args:
        docs (list): A list of documents.

    Returns:
        dict: A dictionary where the keys are document names and the values are
        the concatenated content of the documents.
    """
    content_dict = {}

    for doc in docs:
        doc_name = str(Path(doc.metadata["source"]).resolve())
        doc_content = doc.page_content

        if doc_name not in content_dict:
            content_dict[str(doc_name)] = doc_content
        else:
            content_dict[str(doc_name)] += doc_content

    return content_dict


def load_and_process_pdfs_by_langchain(path: Path) -> dict[str, str]:
    return process_documents_by_langchain(load_documents_by_langchain(path))


def load_and_process_one_pdf_by_azure_document_intelligence(
    path: Path,
    key: str,
    endpoint: str,
) -> str:
    pages = len(PyPDFLoader(str(path)).load())
    document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    with path.open("rb") as file:
        result = document_analysis_client.begin_analyze_document(
            "prebuilt-document",
            file,
            pages=f"1-{pages}",
        ).result()
    return result.content


def load_and_process_pdfs_by_azure_document_intelligence(path: Path) -> dict[str, str]:
    config = Config()

    assert config.azure_document_intelligence_key is not None
    assert config.azure_document_intelligence_endpoint is not None

    content_dict = {}
    ab_path = path.resolve()
    if ab_path.is_file():
        assert ".pdf" in ab_path.suffixes, "The file must be a PDF file."
        proc = load_and_process_one_pdf_by_azure_document_intelligence
        content_dict[str(ab_path)] = proc(
            ab_path,
            config.azure_document_intelligence_key,
            config.azure_document_intelligence_endpoint,
        )
    else:
        for file_path in ab_path.rglob("*"):
            if file_path.is_file() and ".pdf" in file_path.suffixes:
                content_dict[str(file_path)] = load_and_process_one_pdf_by_azure_document_intelligence(
                    file_path,
                    config.azure_document_intelligence_key,
                    config.azure_document_intelligence_endpoint,
                )
    return content_dict
