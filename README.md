# Image RAG Search

## Overview

Image RAG Search is a project that allows users to search for images based on generated descriptions using Retrieval-Augmented Generation (RAG) architecture. The project encodes images from a directory into base64 format, passes them through the Llama 3.2 Vision model to generate descriptions, and implements a RAG architecture to retrieve images based on user queries.

## How It Works

1. **Image Encoding**: Images from a specified directory are encoded into base64 format.
2. **Description Generation**: The base64-encoded images are passed through the Llama 3.2 Vision model to generate structured descriptions.
3. **RAG Architecture**: The generated descriptions are used to create embeddings, which are stored in a FAISS index. The RAG architecture retrieves images based on user queries by searching the FAISS index.

## Project Structure

![Image RAG Architecture](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*NXp5Shd63RKaDxEsEgZw1g.jpeg)

## Usage

### Prerequisites

- Docker
- Docker Compose

### Running the Project

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/yourusername/image-rag-search.git
    cd image-rag-search
    ```

2. **Add groq api to .env**

3. **Add your images to image directory in fast_api**

3. **Build and Start the Docker Containers**:
    ```sh
    docker-compose up --build
    ```

4. **Access the Streamlit Application**:
    Open your web browser and navigate to `http://localhost:8501`.

### Docker Compose Configuration

The [docker-compose.yml]file defines two services: [fastapi] and [streamlit].

- **FastAPI Service**:
    - Builds from the [fast_api](http://_vscodecontentref_/9) directory.
    - Exposes port `8000`.
    - Uses a named volume `fastapi_data` for persistent storage.
    - Mounts the [image_rag](http://_vscodecontentref_/10) directory for image storage.

- **Streamlit Service**:
    - Builds from the [streamlit](http://_vscodecontentref_/11) directory.
    - Depends on the [fastapi](http://_vscodecontentref_/12) service.
    - Exposes port `8501`.

### API Endpoints

- **Generate Descriptions**:
    - Endpoint: `POST /generate_description`
    - Description: Generates image descriptions from the images in the [image_rag](http://_vscodecontentref_/13) directory.

- **Generate Embeddings**:
    - Endpoint: `POST /generate_description_embeddings`
    - Description: Generates embeddings from the image descriptions and stores them in a FAISS index.

- **Search Images**:
    - Endpoint: `POST /get_image`
    - Description: Retrieves images based on the user query by searching the FAISS index.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.
