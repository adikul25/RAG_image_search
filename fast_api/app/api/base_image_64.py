import os
import time
import base64
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

class ImageDescriptionGenerator:
    def __init__(self, directory_path):
        """
        Initializes the ImageDescriptionGenerator class with the image directory path, 
        model details, and API key.
        """
        self.directory_path = directory_path
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = ChatGroq(temperature=0.5, model="llama-3.2-11b-vision-preview", api_key=self.api_key)
    
    def get_images_from_directory(self):
        """Fetch all image file paths from the given directory."""
        valid_extensions = (".jpg", ".jpeg", ".png")  # Allowed image formats
        return [
            os.path.join(self.directory_path, file)
            for file in os.listdir(self.directory_path)
            if file.lower().endswith(valid_extensions)  # Filter only images
        ]
    
    def encode_images_to_base64(self, image_paths):
        """Encodes images to base64 format."""
        encoded_images = []
        for path in image_paths:
            with open(path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                encoded_images.append({"image_path": path, "image_base64": image_base64})
        return encoded_images
    
    def generate_image_descriptions(self):
        """Passes encoded images to the LLM and returns descriptions."""
        image_paths = self.get_images_from_directory()
        encoded_images = self.encode_images_to_base64(image_paths)

        prompt_text = (
        '''
        Analyze the given base64-encoded image and generate a structured description covering the following aspects, the reponse should be stirctly in the format given:

        Output Format:

        General Overview: Provide a high-level summary of the image, describing the primary subject and scene.
        Objects & Entities: List all identifiable objects, people, animals, or elements in the image.
        Clothing & Accessories: Detail any notable clothing, accessories, or personal items visible.
        Actions & Interactions: Identify any activities or interactions occurring within the scene.
        Environment & Background: Describe the setting and contextual details (e.g., indoors, outdoors, cityscape, natural scenery).
        Location: If the location is identifiable based on surroundings (e.g., landmarks, signage, geographic features), mention it. If it cannot be determined, explicitly state that the location is unknown. Do not assume or infer.
        Emotions & Expressions: If people are present, describe their facial expressions, emotions, and possible intent.
        Possible Context & Meaning: Infer the possible context, intent, or message conveyed by the image.


        The output should be structured and include all necessary components for effective RAG search
        Keep the descriptions detailed and precise, don't be overly verbose.
        Keep it in less than **150 words** strictly while covering all the aspects in given format.
        '''
        )

        contextualized_images = []

        for img in encoded_images:
            msg = self.model.invoke([
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img['image_base64']}"}}  
                    ]
                )
            ])
            
            # Create a LangChain Document object
            contextualized_content = msg.content
            contextualized_images.append(Document(page_content=contextualized_content, metadata={"image_path": img["image_path"]}))

            time.sleep(2.5)
        return contextualized_images