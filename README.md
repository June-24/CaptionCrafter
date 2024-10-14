# CaptionCrafter

# Image Caption Generator with Deep Learning and Groq Cloud

## Overview

This project is an **Image Caption Generator** application built using **Streamlit**. It leverages deep learning models to generate captions for images using a local trained model and also utilizes **Groq Cloud** for advanced caption generation. The application aims to provide users with catchy Instagram captions, relevant hashtags, and comments to enhance their social media presence.

## Features

- **Image Upload**: Users can upload images in JPG, JPEG, or PNG formats.
- **Caption Generation**:
  - **Local Model**: Generates captions using a pre-trained Convolutional Neural Network (CNN) combined with Long Short-Term Memory (LSTM) networks trained on the Flickr30k dataset.
  - **Groq Cloud Integration**: Utilizes Groq's advanced API to generate high-quality captions and additional Instagram content.
- **Instagram Content Generation**: Automatically generates catchy captions, hashtags, and comments tailored for Instagram.

## Technologies Used

- **Streamlit**: A Python library for creating web applications.
- **TensorFlow**: For building and training the deep learning models.
- **Keras**: A high-level neural networks API.
- **Groq**: For cloud-based AI capabilities.
- **LangChain**: For interfacing with Groq’s language model.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone [<repository-url>](https://github.com/June-24/CaptionCrafter)
   ```

2. Navigate to the project directory:
   ```bash
   cd <project-directory>
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables in a `secrets.toml` file for the Groq API key (or keep a .env file and store the API key):
   ```toml
   [GROQ]
   API_KEY = "your_api_key_here"
   ```

5. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage

- Upload an image to generate captions.
- Choose whether to use the local model or Groq Cloud for caption generation.
- View the generated caption, hashtags, and a suggested comment for Instagram.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you find any bugs or have suggestions for improvements.

