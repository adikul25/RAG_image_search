import os
import json
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class ImageRetriever:
    def __init__(self):
        """
        Initializes the ImageRetriever class with FAISS index path and model details.
        """
        self.faiss_path = os.getenv("FAISS_PATH")
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = os.getenv("MODEL_NAME")
        self.embedding_model = os.getenv("EMBEDDING_MODEL")

        self.model = ChatGroq(temperature=0.5, model=self.model_name, api_key=self.api_key)
        
        # Load FAISS with the correct embedding function
        if not os.path.exists(self.faiss_path):
            raise ValueError(f"FAISS index path '{self.faiss_path}' does not exist. Run embedding generation first.")

        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.db = FAISS.load_local(self.faiss_path, self.embeddings, allow_dangerous_deserialization=True)

    def retrieve_images(self, query_text, k=5):
        """
        Retrieves the most relevant images based on the query text using FAISS similarity search.
        """
        results = self.db.similarity_search_with_relevance_scores(query_text, k=k)
        image_paths = [doc.metadata.get("image_path", "") for doc, _ in results]
        
        PROMPT_TEMPLATE = """
                        Retrieve the most relevant images based on the given context.
                        You are provided with a set of image file paths along with their associated descriptions generated from a Faiss similarity store.

                          Image Paths:
                          {image_list}

                          Context:
                          {context}

                          ---

                          Based on the above information, select the most relevant image(s) that best match the query. For example, if the query is "get my image in snow wearing something yellow", ensure you only return images whose descriptions align with this query:

                          Query: {question}

                          Return only a valid JSON object with the following format:

                          {{
                              "image_name": ["name1", "name2", "name3", ...]
                          }}

                            Instructions:
                        - Only include image file names that exist in the provided image list.
                        - Do not generate any image names on your own.
                        - Do not add any extra explanations or text.
                        - Limit the returned images to only those that match the query context.
                        - Each image name must appear only **once**. 
                          """

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        image_list = ",\n        ".join(f'"{path}"' for path in image_paths)
        
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text, image_list=image_list)

        response_text = self.model.predict(prompt)
        
        return response_text