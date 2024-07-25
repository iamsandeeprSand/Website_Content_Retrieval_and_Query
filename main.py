import string
import requests
from bs4 import BeautifulSoup
import pandas as pd
import nest_asyncio
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Apply nest_asyncio to allow nested asyncio.run calls
nest_asyncio.apply()

# Web Scraper class definition
class WebScraper:
    def __init__(self, url, headers=None):
        self.url = url

    def extract_paragraphs(self, html_content):
        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')
            paragraphs = [p.text for p in soup.find_all('p')]
            return paragraphs
        else:
            return []

    def fetch_and_extract_p(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            paragraphs = self.extract_paragraphs(response.text)
            return " ".join(paragraphs)
        else:
            return None

# Text preprocessing function
def preprocess_text(sentences):
    # Initialize stop words and lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    preprocessed_sentences = []
    for sentence in sentences:
        # Lowercase the sentence
        sentence = sentence.lower()
        
        # Remove punctuation
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize the sentence
        tokens = word_tokenize(sentence)
        
        # Remove stop words
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatize the tokens
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        preprocessed_sentences.append(" ".join(tokens))

    return preprocessed_sentences

# Function to preprocess paragraphs and split into sentences
def paragraph_to_preprocessed_sentences(paragraph):
    sentences = sent_tokenize(paragraph)
    preprocessed_sentences = preprocess_text(sentences)
    return preprocessed_sentences

# Load sentence transformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to get embeddings
def get_embeddings(sentences):
    embeddings = embedding_model.encode(sentences, show_progress_bar=True)
    return embeddings

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define schema for Milvus collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512)
]
schema = CollectionSchema(fields, description="Text embeddings")

# Create collection
collection_name = "text_embedding_collection"
collection = Collection(name=collection_name, schema=schema)

# Create index
index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 100},
    "metric_type": "L2"
}
collection.create_index(field_name="embedding", index_params=index_params)

def store_in_milvus(sentences, embeddings):
    # Ensure embeddings are in the correct format
    embeddings = embeddings.tolist()
    
    # Truncate sentences to max length of 512 characters
    truncated_sentences = [sentence[:512] for sentence in sentences]
    
    # Prepare entities for Milvus
    entities = {
        "embedding": embeddings,
        "text": truncated_sentences
    }
    
    collection.insert([entities["embedding"], entities["text"]])
    collection.flush()

# Function to fine-tune GPT-2 model
def fine_tune_gpt2(training_data_path):
    # Load pre-trained GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Create a dataset
    def load_dataset(file_path, tokenizer, block_size=128):
        dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=block_size
        )
        return dataset

    # Create a data collator
    def create_data_collator(tokenizer):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        return data_collator

    # Load dataset and data collator
    train_dataset = load_dataset(training_data_path, tokenizer)
    data_collator = create_data_collator(tokenizer)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir='./gpt2_finetuned',
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the model
    trainer.save_model('./gpt2_finetuned')
    tokenizer.save_pretrained('./gpt2_finetuned')

# FastAPI app definition
app = FastAPI()

class LoadRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    query: str
    use_milvus: bool = True

@app.post("/load")
async def load_website_content(request: LoadRequest):
    scraper = WebScraper(request.url)
    text_content = scraper.fetch_and_extract_p()
    if not text_content:
        return {"message": "Failed to fetch or extract content from the URL."}
    
    sentences = paragraph_to_preprocessed_sentences(text_content)
    
    # Create a DataFrame from sentences
    df = pd.DataFrame(sentences, columns=["sentence"])
    training_data_path = "web_content.txt"
    df.to_csv(training_data_path, index=False, header=False)

    # Generate embeddings for the sentences
    embeddings = get_embeddings(sentences)

    # Store in Milvus
    store_in_milvus(sentences, embeddings)

    # Fine-tune GPT-2 model on the fetched data
    fine_tune_gpt2(training_data_path)

    return {"message": "Content loaded, dataset created, embeddings stored, and model fine-tuned successfully"}

@app.post("/query")
async def query_content(request: QueryRequest):
    if request.use_milvus:
        # Generate embedding for the query
        query_embedding = get_embeddings([request.query])[0]
        
        # Fetch from Milvus
        collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=5,
            expr=None
        )
        context = [res.entity.get("text") for res in results[0] if res.entity.get("text") is not None]
        
        # Ensuring the response is limited to the content stored in the database
        df = pd.read_csv("web_content.txt", header=None)
        filtered_context = [text for text in context if text in df[0].values]
    else:
        context = []

    # Generate answer using the fine-tuned model
    gpt2_model = GPT2LMHeadModel.from_pretrained("./gpt2_finetuned")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_finetuned")
    inputs = gpt2_tokenizer.encode(request.query + " " + " ".join(filtered_context), return_tensors="pt")
    outputs = gpt2_model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=3,  # Prevent repeating trigrams
        repetition_penalty=2.0,  # Penalize repeated tokens more heavily
        temperature=0.7,  # Sampling temperature
        top_p=0.9  # Top-p (nucleus) sampling
    )
    answer = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
