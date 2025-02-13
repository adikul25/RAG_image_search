from fastapi import APIRouter
from api.base_image_64 import ImageDescriptionGenerator
from api.generate_embeddings import ImageEmbeddingProcessor
from api.models import ImageDescription, Query
from api.retrieve_image import ImageRetriever
from concurrent.futures import ThreadPoolExecutor
import os

router = APIRouter()

@router.post("/generate_description")
def generate_image_ddscription():
    current_directory = os.getcwd()
    directory_path = os.path.abspath(os.path.join(current_directory, '..', 'app', 'image_rag'))
    path_base64 = ImageDescriptionGenerator(directory_path)
    info = path_base64.generate_image_descriptions()
    return info

@router.post("/generate_description_embeddings")
def generate_description_embeddings(input: ImageDescription):
    response = ImageEmbeddingProcessor()
    documents = response.transform_to_langchain_documents(input.descriptions)
    faiss_db = response.generate_embeddings(documents)
    retrieved_faiss_db = response.load_faiss_index()

@router.post("/get_image")
def search_image(input: Query):
    retriever = ImageRetriever()
    response = retriever.retrieve_images(input.query)
    return response

