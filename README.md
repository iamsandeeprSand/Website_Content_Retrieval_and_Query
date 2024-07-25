# Web Scraping and Text Analysis with GPT-2 Fine-Tuning and Milvus Integration

This project demonstrates a pipeline for web scraping, text preprocessing, embeddings generation, storage in Milvus, and fine-tuning GPT-2 for text generation. It integrates multiple technologies to showcase a complete workflow for handling text data and generating context-aware responses.

## Table of Contents

- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Functions and Classes](#functions-and-classes)
- [Project Structure](#project-structure)
- [Contributing](#contributing)


## Introduction

This project is designed to scrape web content, preprocess the text, generate embeddings, store them in Milvus, and fine-tune a GPT-2 model to generate responses based on the stored text. It demonstrates the integration of various tools and frameworks to build a comprehensive text processing pipeline.

## Technologies Used

- **Web Scraping**: `requests`, `BeautifulSoup`
- **Text Preprocessing**: `nltk`
- **Embeddings**: `SentenceTransformer`
- **Vector Database**: `Milvus`
- **Deep Learning Model**: `GPT-2` from `transformers`
- **Web Framework**: `FastAPI`
- **Containerization**: `Docker`
- **Testing**: `Postman`

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/iamsandeeprSand/Website_Content_Retrieval_and_Query.git
    cd Website_Content_Retrieval_and_Query
    ```

2. **Set up a virtual environment and install dependencies**:
    ```bash
    python -m venv venv
    source venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Run Milvus using Docker**:
    ```bash
    docker pull milvusdb/milvus
    docker run -d --name milvus_cpu_8000 -p 19530:19530 -p 19121:19121 milvusdb/milvus:latest
    ```

4. **Run the FastAPI application**:
    ```bash
    uvicorn main:app --reload
    ```

## Usage

1. **Loading Website Content**:
    Use Postman to send a POST request to `http://127.0.0.1:8000/load` with a JSON body containing the URL to scrape:
    ```json
    {
        "url": "https://en.wikipedia.org/wiki/Large_language_model"
    }
    ```

2. **Querying Content**:
    Use Postman to send a POST request to `http://127.0.0.1:8000/query` with a JSON body containing the query:
    ```json
    {
        "query": "What is Large Language Model?",
        "use_milvus": true
    }
    ```

## API Endpoints

- **`POST /load`**: Scrapes content from the provided URL, preprocesses the text, generates embeddings, stores them in Milvus, and fine-tunes a GPT-2 model.
    - **Request Body**: `{ "url": "<URL to scrape>" }`
    - **Response**: `{ "message": "Content loaded, dataset created, embeddings stored, and model fine-tuned successfully" }`

- **`POST /query`**: Queries the stored content and generates a response using the fine-tuned GPT-2 model.
    - **Request Body**: `{ "query": "<query>", "use_milvus": <boolean> }`
    - **Response**: `{ "answer": "<generated answer>" }`

## Functions and Classes

### Classes

- **`WebScraper`**: 
  - `__init__(self, url, headers=None)`: Initializes the scraper with a URL.
  - `extract_paragraphs(self, html_content)`: Extracts paragraphs from the HTML content.
  - `fetch_and_extract_p(self)`: Fetches the web page content and extracts paragraphs.

### Functions

- **`preprocess_text(sentences)`**: Preprocesses a list of sentences (lowercasing, removing punctuation, tokenizing, removing stop words, and lemmatizing).
- **`paragraph_to_preprocessed_sentences(paragraph)`**: Converts a paragraph to a list of preprocessed sentences.
- **`get_embeddings(sentences)`**: Generates embeddings for a list of sentences using a sentence transformer model.
- **`store_in_milvus(sentences, embeddings)`**: Stores sentences and their embeddings in a Milvus collection.
- **`fine_tune_gpt2(training_data_path)`**: Fine-tunes a GPT-2 model using the text data at the specified path.

## Project Structure

```
.
├── main.py                  # Main FastAPI application
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── web_content.txt          # Fetched and preprocessed web content
```

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.
