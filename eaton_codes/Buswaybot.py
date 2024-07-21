#Buswaybot.py
#import ollama
import streamlit as st
import argparse
from query import query_rag
#import subprocess

####################################################################
#            Create app interface with streamlit
####################################################################


st.set_page_config(
    page_title="Welcome to the 🤖 Eaton Busway chatbot",
    layout="wide",

)

def main():
    st.title("🤖 Busway chatbot")
    st.write("""
    This app allows you to ask questions and get answers based on Eaton Test Data.
    """)

    # Initialize session state if 'messages' doesn't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    message_container = st.container(height=500, border=True)

    # Display existing messages in the session state
    for message in st.session_state["messages"]:
        avatar = "🤖" if message["role"] == "assistant" else "😎"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Input prompt from the user
    prompt = st.text_input("Enter your query:")

    if st.button("Submit"):
        if prompt:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            message_container.chat_message("user", avatar="😎").markdown(prompt)

            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner(":green[processing...]"):
                    try:
                        response = query_rag(prompt)
                        st.session_state["messages"].append({"role": "assistant", "content": response})
                        st.markdown(response)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()

