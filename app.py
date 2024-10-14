import streamlit as st
import numpy as np
import requests
import base64
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import DenseNet201
import pickle
from groq import Groq
from langchain_groq import ChatGroq
import os
# from dotenv import load_dotenv
# Load environment variables from .env file
# load_dotenv()

# Load the API key from secrets
groq_api_key = st.secrets["GROQ"]["API_KEY"]

# Load the tokenizer and local model (already saved)
@st.cache_resource
def load_resources():
    # Load tokenizer
    with open('tokenizer.pkl', 'rb') as file:
        saved_tokenizer = pickle.load(file)
    
    # Load the saved captioning model
    saved_model = load_model('model.keras')
    
    # Load DenseNet201 model for feature extraction
    base_model = DenseNet201()
    fe = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    
    return saved_tokenizer, saved_model, fe

# Utility functions
def idx_to_word(integer, tokenizer):
    """Convert an index to a word using the tokenizer."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, tokenizer, max_length, feature):
    """Generate a caption for an image feature using the local model."""
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]  # Convert to sequence
        sequence = pad_sequences([sequence], maxlen=max_length)  # Pad the sequence

        # Predict next word
        y_pred = model.predict([feature, sequence], verbose=0)
        y_pred = np.argmax(y_pred)  # Get index of the highest probability word
        
        word = idx_to_word(y_pred, tokenizer)  # Get the corresponding word
        
        if word is None:
            break
        
        in_text += " " + word  # Append word to the input text
        
        if word == 'endseq':
            break
            
    return in_text.replace('startseq', '').replace('endseq', '').strip()

def encode_image(image_file):
    """Encode image to base64."""
    return base64.b64encode(image_file.read()).decode('utf-8')

def generate_caption_groq_cloud(api_key, base64_image):
    """Send a request to Groq Cloud API to generate a caption, hashtags, and a comment for an image."""
    
    # Ensure the API key is passed to the Groq client
    client = Groq(api_key=api_key)
    
    # Prepare the API request
    completion = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": """ (NO PREAMBLE)Generate a catchy caption for this image that is easy to understand. 
                        Then generate hashtags under the caption to make the post more popular. 
                        Lastly, generate the first comment that I should post as the owner of the image.
                        Format the output in the following way:
                        
                        ### Caption
                        <Caption text here>
                        
                        ### Hashtags
                        <Hashtags here>
                        
                        ### First Comment
                        <Comment here>"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
    )
    
    return completion.choices[0].message.content



def generate_instagram_content_groq(api_key, caption):
    """Use Groq's LLM to generate Instagram hashtags and comments."""
    
    # Call the text generation model for Instagram content
    llm = ChatGroq(
        temperature=0.7, 
        groq_api_key=api_key, 
        model_name="llama-3.1-70b-versatile"
    )

                        
    prompt = f"(NO PREAMBLE)generate hashtags under the caption to make the post more popular. Lastly, generate the first comment that I should post as the owner of the image. Format the output in the following way: ### Hashtags  <Hashtags here> ### First Comment  <Comment here>        caption: '{caption}'."
                             
    response = llm.invoke(prompt)
    return response.content


st.set_page_config(
    page_title="CaptionCrafter",  # Change this to your preferred title
    page_icon="📸",  # Optional: You can set an emoji or path to an icon file
    layout="wide"  # Optional: Set to 'centered' or 'wide' depending on your layout preference
)

# Main Streamlit App
def main():
    st.write("## CaptionCrafter: From Images to Insights")
    st.write("This app generates a caption for an image using a local pretrained model (CNN + LSTM) or Groq Cloud and also generates Instagram content like hashtags and comments.")

    # API Key input for Groq Cloud service
    use_groq_cloud = st.checkbox("Use Groq Cloud for caption generation")
    # groq_api_key = os.getenv("GROQ_API_KEY")

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Split layout into two columns (left: image, right: caption)
        col1, col2 = st.columns([1, 2])  # Adjust the proportions if needed
        
        with col1:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            if use_groq_cloud and groq_api_key:
                # Encode the image as base64
                base64_image = encode_image(uploaded_file)
                
                # Generate caption and Instagram content using Groq Cloud
                st.write("Using Groq Cloud to generate caption and Instagram content...")
                caption = generate_caption_groq_cloud(groq_api_key, base64_image)
                
                # Display the generated caption and Instagram content
                st.write("### Generated Instagram Caption & Hashtags:")
                st.write(caption)

            else:
                # Load the resources
                tokenizer, model, fe = load_resources()

                # Process the uploaded image
                img = load_img(uploaded_file, target_size=(224, 224))
                img = img_to_array(img)
                img = img / 255.0
                img = np.expand_dims(img, axis=0)  # Add batch dimension

                # Extract features using DenseNet201
                feature = fe.predict(img, verbose=0)

                # Generate caption locally
                max_length = 74  # Set max_length derived from training
                caption = predict_caption(model, tokenizer, max_length, feature)

                if groq_api_key:
                    # Generate Instagram content using Groq Cloud's text model
                    st.write("Generating Instagram content...")
                    instagram_content = generate_instagram_content_groq(groq_api_key, caption)
                else:
                    instagram_content = "No Instagram content available without Groq API key."

                # Display the generated caption and Instagram content
                st.write("### Generated Caption:")
                st.write(caption)
                st.write("### Generated Instagram Caption & Hashtags:")
                st.write(instagram_content)


if __name__ == '__main__':
    main()
