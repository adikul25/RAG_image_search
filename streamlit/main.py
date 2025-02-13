import streamlit as st
import requests
import json

# Set the page configuration for a better layout
st.set_page_config(page_title="RAG Image Search", layout="wide")

# Sidebar for additional information or options
st.sidebar.write("### RAG Image Search")
st.sidebar.write("""
    This app allows you to search for images based on generated descriptions 
    using RAG (Retrieval-Augmented Generation) architecture.
    - **Generate Embeddings**: Generate image descriptions and embeddings.
    - **Search**: Enter a query to retrieve images based on the description.
""")

# Main title of the app
st.title("RAG Image Search")

# Button to generate embeddings
if st.button("Generate Embeddings"):
    try:
        # Send request to generate descriptions
        response = requests.post("http://fastapi:8000/generate_description")
        response.raise_for_status()  # Raises an exception for HTTP errors

        # Process the description
        descriptions = response.json()
        input_data = {"descriptions": descriptions}
        
        # Generate the embeddings
        create_vector_store = requests.post("http://fastapi:8000/generate_description_embeddings", json=input_data)
        create_vector_store.raise_for_status()  # Raises an exception for HTTP errors

        st.success("Embeddings generated successfully!")
    except requests.exceptions.RequestException as e:
        st.error(f"Error generating embeddings: {e}")

# Add a text input field for the query
query = st.text_input("Enter your question:")

# Button to trigger the search
if st.button("Search"):
    if query:
        try:
            # Send request to get images based on the query
            search_response = requests.post("http://fastapi:8000/get_image", json={"query": query})
            search_response.raise_for_status()  # Raises an exception for HTTP errors

            # Display the images retrieved from the search
            data = json.loads(search_response.json())
            image_paths = data.get("image_name", [])
            tabs = st.tabs([f"Image {i+1}" for i in range(len(image_paths))])

            for i, img in enumerate(image_paths):
                with tabs[i]:
                    st.image(f"http://localhost:8000/images/{img}", width=300,output_format="auto")

        except requests.exceptions.RequestException as e:
            st.error(f"Error searching for images: {e}")
    else:
        st.warning("Please enter a question before searching.")
