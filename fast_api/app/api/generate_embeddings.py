import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document  # Import Document class from LangChain

class ImageEmbeddingProcessor:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2", faiss_path="faiss_index"):
        """
        Initializes the class with an embedding model and FAISS storage path.
        """
    
        self.embedding_model = embedding_model
        self.faiss_path = faiss_path
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        
        # Ensure FAISS directory exists
        if not os.path.exists(self.faiss_path):
            os.makedirs(self.faiss_path)

    def transform_to_langchain_documents(self, json_data):
        """
        Transforms a list of JSON data into LangChain Document objects.
        """
        documents = []
        for img in json_data:
            doc = Document(
                page_content=img["page_content"],
                metadata={"image_path": img["metadata"]["image_path"]}
            )
            documents.append(doc)
        return documents

    def split_documents(self, documents):
        """
        Splits documents into smaller chunks for better embedding efficiency.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True
        )
        return text_splitter.split_documents(documents)

    def generate_embeddings(self, contextualized_documents):
        """
        Processes contextualized image descriptions, splits them, and stores embeddings in FAISS.
        """
        # Step 1: Split documents into chunks
        chunks = self.split_documents(contextualized_documents)
        print(f"Split {len(contextualized_documents)} documents into {len(chunks)} chunks.")

        # Step 2: Store embeddings in FAISS
        db = FAISS.from_documents(chunks, self.embeddings)
        
        # Save FAISS index
        db.save_local(self.faiss_path)
        print(f"Saved {len(chunks)} chunks to {self.faiss_path}.")
        
        return db  # Returning the FAISS database instance for retrieval if needed

    def load_faiss_index(self):
        """
        Loads the FAISS index from the saved directory.
        """
        if os.path.exists(self.faiss_path):
            return FAISS.load_local(self.faiss_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            print("FAISS index not found. Returning None.")
            return None
