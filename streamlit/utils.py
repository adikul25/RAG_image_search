import json
import streamlit as st


def display_images(response_text):
    data = json.loads(response_text)
    image_paths = data.get("image_name", [])

    tabs = st.tabs([f"Image {i+1}" for i in range(len(image_paths))])

    for i, img in enumerate(image_paths):
        with tabs[i]:
            st.image(f"http://fastapi:8000/images/{img}", width=300,output_format="auto")
