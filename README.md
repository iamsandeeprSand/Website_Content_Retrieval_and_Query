# Web Content Retrieval and Query System

This project is a web-based system that retrieves content from a specified URL, processes it into sentences, embeds those sentences, stores the embeddings in Milvus, and allows querying for relevant content using a GPT-2 model to generate answers. 

## Features

- **Web Scraping**: Fetch and extract text content from web pages.
- **Sentence Embeddings**: Generate embeddings for sentences using a pre-trained sentence transformer model.
- **Milvus Integration**: Store and search sentence embeddings in a Milvus collection.
- **GPT-2 Model**: Generate answers to queries using the GPT-2 language model.
- **FastAPI Backend**: Serve the functionalities via a FastAPI application.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Milvus (Follow the [Milvus installation guide](https://milvus.io/docs/v2.0.0/install_standalone-docker.md) to set up Milvus)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/iamsandeeprSand/Website_Content_Retrieval_and_Query.git
   cd Website_Content_Retrieval_and_Query
   
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   

## Running the Application
1. Ensure Milvus is running on your machine or server.
2. Start the FastAPI server
   ```bash
   python main.py
3. The application will be accessible at http://127.0.0.1:8000.

API Endpoints
Load Website Content
Endpoint: /load
Method: POST
   Request Body
   
               {
               "url": "https://example.com"
               }

   Response:
     
     {
       "message": "Content loaded successfully"
     }


Query Content
Endpoint: /query
Method: POST
   Request Body:
   
     ```bash
     {
       "query": "What is the capital of France?",
       "use_milvus": true
     }
   Response:
     
     {
       "answer": "The capital of France is Paris."
     }
Deployment
   Local Deployment
   To run the application locally, ensure you have followed the setup instructions and installed all necessary packages. Use the command:
   
     ```bash
     uvicorn main:app --host 127.0.0.1 --port 8000

# Detailed Function Descriptions
## WebScraper Class
**__init__(self, url, headers=None):**  Initializes the WebScraper with the given URL.
**extract_paragraphs(html_content):** Extracts paragraphs from the HTML content.
**fetch_page(self):** Fetches the HTML content of the page.
**fetch_and_extract_p(self):** Fetches the HTML content and extracts paragraphs, returning them as a single string.

## Utility Functions
**paragraph_to_sentences(paragraph):** Splits a paragraph into sentences and tokenizes them into words.
**get_embeddings(sentences):** Generates embeddings for a list of sentences using the pre-trained sentence transformer model.
**store_in_milvus(sentences, embeddings):** Stores sentences and their embeddings in Milvus.
**fetch_from_milvus(query_embedding, top_k=5):** Fetches the top-k most similar sentences from Milvus based on the query embedding.
**generate_answer(query, context, max_length=100):** Generates an answer to the query using GPT-2, with the provided context.

# Models Used
Sentence Transformer Model
**Model:** all-MiniLM-L6-v2
**Purpose:** To generate sentence embeddings for the retrieved text content.
**Details:** This model is a smaller, faster, and more efficient version of the BERT model, designed for sentence and paragraph embeddings.
GPT-2 Model
**Model:** gpt2
**Purpose:** To generate natural language answers based on the query and context.
**Details:** GPT-2 is a large transformer-based language model trained by OpenAI that can generate coherent and contextually relevant text.
Packages and Their Uses
**beautifulsoup4:** For parsing HTML content and extracting text.
**requests:** For making HTTP requests to fetch web pages.
**Flask:** A lightweight WSGI web application framework (used in the example code but not in the FastAPI-based implementation).
**nltk:** For natural language processing tasks such as sentence tokenization.
**sentence-transformers:** For generating sentence embeddings using pre-trained models.
**transformers:** For utilizing the GPT-2 model to generate text.
**pymilvus:** For interacting with the Milvus vector database to store and retrieve embeddings.
**nest_asyncio:** To allow nested use of asyncio.run() (useful in Jupyter notebooks).
**uvicorn:** An ASGI server for serving the FastAPI application.
**fastapi:** A modern, fast web framework for building APIs with Python.
**pydantic:** For data validation and settings management using Python type annotations.

  
## Docker Deployment
   You can use Docker to containerize the application for easier deployment. Create a Dockerfile with the following contents:
      ```bash
   # Use the official Python image from the Docker Hub
         FROM python:3.8-slim
        
        # Set the working directory
        WORKDIR /app
        
        # Copy the requirements.txt file
        COPY requirements.txt .
        
        # Install the dependencies
        RUN pip install --no-cache-dir -r requirements.txt
        
        # Copy the application code
        COPY . .
        
        # Expose the port the app runs on
        EXPOSE 8000
        
        # Run the application
        CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

   To build and run the Docker container:
      
         docker build -t web-content-query-system
         docker run -p 8000:8000 web-content-query-system

## Cloud Deployment
For deploying on a cloud platform like AWS, Google Cloud, or Azure, follow the respective platform's instructions to deploy a Docker container. Ensure you have set up the necessary environment variables and configurations.

## Presentation and Architecture
A detailed presentation of the system architecture, design decisions, and implementation steps can be found in the docs directory. The presentation includes diagrams, flowcharts, and explanations of key components.

## Working Demo
To see the system in action, follow the setup instructions and run the application locally or deploy it to your preferred platform. A demonstration video is available in the docs/demo.mp4 file.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.
 












