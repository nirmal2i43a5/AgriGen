from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader


def load_all_documents(data_dir: str) -> List[Any]:

    
    data_path = Path(data_dir).resolve()
    print(f"Loading documents from: {data_path}")
    documents = []

    # PDF files
    for pdf_file in data_path.glob('**/*.pdf'):
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())
            print(f"Loaded PDF: {pdf_file.name}")
        except Exception as e:
            print(f"Failed to load PDF {pdf_file}: {e}")

    # TXT files
    for txt_file in data_path.glob('**/*.txt'):
        try:
            loader = TextLoader(str(txt_file))
            documents.extend(loader.load())
            print(f"Loaded TXT: {txt_file.name}")
        except Exception as e:
            print(f"Failed to load TXT {txt_file}: {e}")

    # CSV files
    for csv_file in data_path.glob('**/*.csv'):
        try:
            loader = CSVLoader(str(csv_file))
            documents.extend(loader.load())
            print(f"Loaded CSV: {csv_file.name}")
        except Exception as e:
            print(f"Failed to load CSV {csv_file}: {e}")

    # Excel files
    for xlsx_file in data_path.glob('**/*.xlsx'):
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            documents.extend(loader.load())
            print(f"Loaded Excel: {xlsx_file.name}")
        except Exception as e:
            print(f"Failed to load Excel {xlsx_file}: {e}")

    # Word files
    for docx_file in data_path.glob('**/*.docx'):
        try:
            loader = Docx2txtLoader(str(docx_file))
            documents.extend(loader.load())
            print(f"Loaded Word: {docx_file.name}")
        except Exception as e:
            print(f"Failed to load Word {docx_file}: {e}")

    # JSON files
    for json_file in data_path.glob('**/*.json'):
        try:
            loader = JSONLoader(str(json_file), jq_schema='.', text_content=False)
            documents.extend(loader.load())
            print(f"Loaded JSON: {json_file.name}")
        except Exception as e:
            print(f"Failed to load JSON {json_file}: {e}")

    print(f"Total loaded documents: {len(documents)}")
    return documents
