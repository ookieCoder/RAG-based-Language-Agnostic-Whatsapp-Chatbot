import os, json, hashlib, torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredPDFLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader, UnstructuredImageLoader

PROCESSED_FILE = "processed_files.json"
DATA_DIR = "data"
VECTOR_DIR = "vector_store"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectordb = Chroma(
    persist_directory=VECTOR_DIR,
    embedding_function=embeddings
)

# Compute file hash (for detecting updates)
def compute_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Load already processed file hashes
def load_processed_files():
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE, "r") as f:
            content = f.read().strip()
            if not content:
                print("\tProcessed files record is empty. Starting fresh.")
                return {}
            return json.loads(content)
    return {}

# Find new or modified files
def get_new_files():
    processed = load_processed_files()
    current_files = {}

    new_files = []
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if not os.path.isfile(path):
            continue

        file_hash = compute_file_hash(path)
        current_files[file] = file_hash

        # Check if new or modified
        if file not in processed or processed[file] != file_hash:
            new_files.append(file)

    return new_files, processed, current_files


# Embed only new files and add to ChromaDB
def update_vector_store():
    new_files, old_records, new_records = get_new_files()

    if not new_files:
        print("\tNo new or updated files detected.")
        return

    print(f"\tFound {len(new_files)} new/updated file(s): {new_files}")

    # Load only new documents
    docs = []
    for file in new_files:
        path = os.path.join(DATA_DIR, file)
        ext = file.lower().split('.')[-1]

        try:
            if ext == "txt":
                loader = TextLoader(path)
            elif ext == "pdf":
                try:
                    loader = PyPDFLoader(path)
                    docs.extend(loader.load())
                    continue
                except:
                    loader = UnstructuredPDFLoader(path, mode="elements")
            elif ext == "docx":
                loader = UnstructuredWordDocumentLoader(path)
            elif ext == "xlsx":
                loader = UnstructuredExcelLoader(path)
            elif ext in ["jpg", "jpeg", "png"]:
                loader = UnstructuredImageLoader(path)
            else:
                print(f"\tSkipping unsupported file: {file}")
                continue

            docs.extend(loader.load())
        except Exception as e:
            print(f"\tError loading {file}: {e}")

    if not docs:
        print("\tNo new documents were successfully loaded.")
        return

    # Create embeddings only for new documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    chunks = splitter.split_documents(docs)
    vectordb.add_documents(chunks)

    # Update record
    old_records.update(new_records)
    with open(PROCESSED_FILE, "w") as f:
        json.dump(old_records, f, indent=4)

    print(f"\tAdded {len(chunks)} chunks from {len(new_files)} file(s) to vector store.")

if __name__ == "__main__":
    print("\tChecking for new files")
    update_vector_store()
