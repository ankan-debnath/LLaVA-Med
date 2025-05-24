import streamlit as st
import requests
from langchain.schema import AIMessage, HumanMessage
import json

st.title(':rainbow[Covid-19 Bot]')

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# To reset uploader, track a dummy key
if "image_uploader_key" not in st.session_state:
    st.session_state.image_uploader_key = 0

# Show all previous chat messages
for message in st.session_state.chat_history:
    with st.chat_message('user' if isinstance(message, HumanMessage) else 'assistant'):
        st.write(message.content)

# Chat input
text = st.chat_input("Ask ChatBot anything!")

# Image uploader below chat input
uploaded_image = st.file_uploader(
    label="📎 Upload an image (optional)",
    type=["jpg", "jpeg", "png"],
    key=f"image_uploader_{st.session_state.image_uploader_key}",
    label_visibility="visible"
)

if text:
    st.session_state.chat_history.append(HumanMessage(content=text))

    chat_history = [
        {'role': 'user' if isinstance(msg, HumanMessage) else 'assistant', 'content': msg.content}
        for msg in st.session_state.chat_history
    ]

    with st.chat_message('user'):
        st.write(text)

    try:
        data = {
            'message': text,
            'chat_history': json.dumps(chat_history)
        }

        files = {'image': uploaded_image} if uploaded_image else None

        response = requests.post(
            'http://127.0.0.1:5000/response',
            data=data,
            files=files
        )

        if response.status_code == 200:
            reply = response.json().get('response', 'No response received')
            st.session_state.chat_history.append(AIMessage(content=reply))
        else:
            reply = "Error: Unable to process request"

    except requests.exceptions.RequestException as e:
        reply = f"Request Failed: {e}"

    with st.chat_message('assistant'):
        st.write(reply)

    # Reset uploader by changing key (forces Streamlit to rerender the uploader as empty)
    st.session_state.image_uploader_key += 1
