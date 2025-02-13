from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from api.router import api_router

app = FastAPI()


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin (change this for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Ensure that FastAPI serves the images
# image_directory = "app/image_rag"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "image_rag")

# if not os.path.exists(image_directory):
#     os.makedirs(image_directory)

app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

app.include_router(api_router)