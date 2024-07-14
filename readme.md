# Chatgroq With Llama3 Demo

## Overview

This application demonstrates the use of the ChatGroq API with Llama3-8b-8192 model for answering questions based on provided document contexts. The application uses Streamlit for the web interface and integrates with Langchain for document processing and embedding.

## Requirements

- Python 3.7+
- Streamlit
- Langchain
- HuggingFace Embeddings
- FAISS
- PyPDFLoader
- dotenv

## Installation

1. Clone the repository:

```sh
git clone <repository_url>
cd <repository_directory>
```

2. Install the required dependencies:

```sh
pip install streamlit langchain huggingface_hub faiss-gpu pypdf2 python-dotenv
```

3. Set up your environment variables. Create a `.env` file in the root directory and add your GROQ API key:

```env
GROQ_API_KEY=your_groq_api_key
```

## Usage

1. Ensure you have the necessary API keys set in your `.env` file.

2. Run the Streamlit application:

```sh
streamlit run app.py
```

3. Upload your PDF documents using the file uploader. The application will process and embed the documents.

4. Enter your questions in the text input field and click "Generate Response" to get answers based on the uploaded documents.

## File Structure

- `app.py`: Main application file
- `.env`: Environment variable file containing API keys
- `requirements.txt`: List of required dependencies

## How It Works

1. **File Upload**: Users can upload multiple PDF documents. The application saves these documents to a temporary directory.

2. **Document Processing**: The application uses PyPDFLoader to read and load the content of the PDF documents. 

3. **Embedding**: The loaded documents are split into smaller chunks using RecursiveCharacterTextSplitter and then embedded using HuggingFaceEmbeddings. The embeddings are stored in a FAISS vector store.

4. **Question Answering**: When a user inputs a question, the application retrieves relevant document chunks from the vector store using a retriever. The retrieved documents are then passed to the ChatGroq model for generating a response.

5. **Response Display**: The generated response along with the relevant document chunks are displayed on the Streamlit interface.

## Additional Information

- **Vector Store**: The FAISS vector store is used to efficiently retrieve relevant document chunks based on the user's question.
- **Model**: The ChatGroq API with Llama3-8b-8192 model is used for generating answers.

## Credits

This application uses the following libraries and services:
- [Streamlit](https://streamlit.io/)
- [Langchain](https://www.langchain.com/)
- [HuggingFace](https://huggingface.co/)
- [FAISS](https://faiss.ai/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [GROQ API](https://www.groq.com/)

Feel free to contribute or raise issues if you find any bugs or have suggestions for improvements.