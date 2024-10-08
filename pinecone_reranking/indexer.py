import os
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import PyPDF2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

open_ai_api_key = "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"

print("Initializing Pinecone...")
pc = Pinecone(api_key='6c634c24-f8cb-44a0-829f-00532451a537')

index_name = "polyloop-index"
if index_name not in pc.list_indexes().names():
    print(f"Creating new Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embeddings have 1536 dimensions
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
else:
    print(f"Using existing Pinecone index: {index_name}")

# Connect to the index
index = pc.Index(index_name)

print("Initializing OpenAI client...")
client = OpenAI(api_key=open_ai_api_key)

# Function to get embeddings using OpenAI
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Function to extract text from a PDF file using PyPDF2
def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from {pdf_path}")
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    print(f"Extracted {len(text)} characters from the PDF")
    return text

# Function to split text into chunks of specified token size
def chunk_text(text, chunk_size=384):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    print(f"Text split into {len(chunks)} chunks")
    return chunks

# Function to generate embeddings in parallel
def generate_embeddings_parallel(chunks):
    embeddings = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_chunk = {executor.submit(get_embedding, chunk): chunk for chunk in chunks}
        for i, future in enumerate(as_completed(future_to_chunk), 1):
            embedding = future.result()
            embeddings.append(embedding)
            if i % 100 == 0:
                print(f"Generated {i} embeddings")
    return embeddings

# Function to upsert vectors in batches
def upsert_in_batches(vectors, batch_size=100):
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
        print(f"Upserted batch {i//batch_size + 1} of {len(vectors)//batch_size + 1}")

# Folder where PDFs are stored
pdf_folder = "/Users/fortanpireva/Projects/Github/ai_research/dspy_llamaparse_big_file/pinecone_reranking/pdf_files"
print(f"Processing PDF files in folder: {pdf_folder}")

# Loop through all the PDF files in the folder
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf") and  not f.startswith("EN")]
total_files = len(pdf_files)
print(f"Total files: {total_files}")

for i, pdf_file in enumerate(pdf_files, 1):
    print(f"\nProcessing file {i} of {total_files}: {pdf_file}")
    pdf_path = os.path.join(pdf_folder, pdf_file)
    
    # Extract text from the PDF using PyPDF2
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Split the text into smaller chunks for embedding
    chunks = chunk_text(pdf_text)
    
    # Generate embeddings for each chunk using OpenAI in parallel
    print("Generating embeddings...")
    chunk_embeddings = generate_embeddings_parallel(chunks)
    print(f"Generated {len(chunk_embeddings)} embeddings in total")
    
    # Prepare data for upsert
    vectors = [
        (f"{pdf_file}_chunk_{i}", 
         chunk_embeddings[i], 
         {"pdf_name": os.path.basename(pdf_path), "chunk_id": i, "chunk_text": chunk})
        for i, chunk in enumerate(chunks)
    ]
    
    # Upsert vectors into Pinecone in batches
    print(f"Upserting {len(vectors)} vectors to Pinecone in batches...")
    upsert_in_batches(vectors)
    print(f"Finished processing {pdf_file}")

print("\nAll PDF files have been successfully indexed with embeddings and metadata.")
print(f"Total files processed: {total_files}")