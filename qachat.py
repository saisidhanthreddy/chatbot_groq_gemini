import streamlit as st
import os
import groq
import google.generativeai as genai
import re  

# Configure page (Must be the first Streamlit command)
st.set_page_config(page_title="AI Travel Recommendation", layout="wide")

# API Key handling in sidebar
with st.sidebar:
    st.title("Configuration")
    
    # LLM Provider Selection
    provider = st.radio("LLM Provider", ["Gemini", "Groq"])
    
    if provider == "Gemini":
        google_api_key = st.text_input("Google API Key", type="password", help="Enter your Google API key here")
        model = st.selectbox("Model", ["gemini-1.5-pro"])
        
        if not google_api_key:
            st.warning("Please enter your Google API key to continue.")

    elif provider == "Groq":
        groq_api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key here")
        model = st.selectbox("Model", ["llama-3.3-70b-versatile", "gemma2-9b-it", "qwen-2.5-32b", "mistral-saba-24b", "deepseek-r1-distill-qwen-32b"])
        
        if not groq_api_key:
            st.warning("Please enter your Groq API key to continue.") 
    st.divider()
    st.write("## About")
    st.write("This app helps you plan your perfect travel itinerary by generating detailed travel guides and recommendations through a multi-step workflow.")
    st.write("Made with ❤️ using LangGraph and Streamlit to make your travel planning easier!")

# Function to get LLM instance with explicit API key
def get_llm(provider, model, api_key=None):
    if provider == "Gemini":
        if not api_key:
            st.error("Please enter your Google API key in the sidebar.")
            st.stop()
        genai.configure(api_key=api_key)  # Configure with the provided key directly
        return genai.GenerativeModel(model).start_chat(history=[])
    
    elif provider == "Groq":
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar.")
            st.stop()
        return groq.Groq(api_key=api_key)  # Pass the key directly

# Ensure chat history is initialized
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'trip_details' not in st.session_state:
    st.session_state['trip_details'] = {}

# Function to classify a sentence as travel-related or not
def classify_query_with_llm(sentence, provider, model, api_key):
    classification_prompt = f"""
    Determine if the following sentence is strictly related to travel, itinerary planning, flights, hotels, or sightseeing.
    Respond with ONLY "Yes" if it is strictly travel-related and "No" if it is not.
    
    Sentence: "{sentence}"
    """
    llm = get_llm(provider, model, api_key)
    
    if provider == "Gemini":
        response = llm.send_message(classification_prompt, stream=False)
        return response.text.strip().lower() == "yes"
    else:  # Groq
        response = llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip().lower() == "yes"

# Function to generate response
def get_llm_response(question, provider, model, api_key):
    llm = get_llm(provider, model, api_key)
    
    if provider == "Gemini":
        response = llm.send_message(question, stream=False)
        response_text = response.text
    else:  # Groq
        response = llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=0.7
        )
        response_text = response.choices[0].message.content.strip()

    # Store in chat history
    st.session_state['chat_history'].append({"role": "assistant", "content": response_text})
    return response_text

# Streamlit chatbot UI
st.title("✈️ AI Travel Chatbot")
st.write("Hello! I can help you plan your trips, find hotels, suggest places to visit, and more. Ask me anything about travel!")

# Chat input
user_input = st.chat_input("Type your travel question here...")

if user_input:
    # Get API key based on provider
    api_key = google_api_key if provider == "Gemini" else groq_api_key
    
    # Process travel-related queries
    if 'trip_details' not in st.session_state or not st.session_state['trip_details']:
        st.session_state['chat_history'].append({"role": "user", "content": user_input})
        travel_query = f"Can you help plan a trip for: {user_input}?"
    else:
        travel_query = f"Continue planning the trip with these details: {st.session_state['trip_details']} and this query: {user_input}"

    # Extract and classify the query as travel-related or not
    if classify_query_with_llm(user_input, provider, model, api_key):
        response = get_llm_response(travel_query, provider, model, api_key)

        # Save trip details if new details are provided
        if "trip" in user_input.lower():
            st.session_state['trip_details'] = user_input

        # Display user input
        with st.chat_message("user"):
            st.write(f"**You:** {user_input}")
        
        # Display response
        with st.chat_message("assistant"):
            st.write(response)
    else:
        error_message = "❌ I can only answer travel-related questions. Please ask about flights, hotels, sightseeing, or trip planning."
        if not any(msg["content"] == error_message for msg in st.session_state['chat_history']):
            st.session_state['chat_history'].append({"role": "assistant", "content": error_message})

# Display chat history
for message in st.session_state['chat_history']:
    with st.chat_message("assistant" if message["role"] != "user" else "user"):
        st.write(f"**{message['role']}:** {message['content']}")