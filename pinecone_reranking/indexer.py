import fitz  # PyMuPDF for PDF text extraction
import os
import pinecone
from sentence_transformers import SentenceTransformer

# Initialize Pinecone
pinecone.init(api_key='YOUR_PINECONE_API_KEY', environment='us-west1-gcp')
index_name = "polyloop-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)  # Assuming the model generates embeddings with dimension 384
index = pinecone.Index(index_name)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text("text")
    return text

# Function to split text into chunks of specified token size
def chunk_text(text, chunk_size=512):
    """Split the extracted text into smaller chunks of specified size."""
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Folder where PDFs are stored
pdf_folder = "/Users/fortanpireva/Projects/Github/ai_research/dspy_llamaparse_big_file/pinecone_reranking/pdf_files"

# Loop through all the PDF files in the folder
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):  # Check for PDF files only
        pdf_path = os.path.join(pdf_folder, pdf_file)
        
        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(pdf_path)
        
        # Split the text into smaller chunks for embedding
        chunks = chunk_text(pdf_text)
        
        # Generate embeddings for each chunk
        chunk_embeddings = model.encode(chunks)
        
        # Prepare metadata (include chunk text and other info)
        metadata_list = [{"pdf_name": os.path.basename(pdf_path), 
                          "chunk_id": i, 
                          "chunk_text": chunk} for i, chunk in enumerate(chunks)]
        
        # Prepare data for upsert: (id, vector, metadata)
        vectors = [(f"{pdf_file}_chunk_{i}", chunk_embeddings[i], metadata_list[i]) for i in range(len(chunk_embeddings))]
        
        # Upsert vectors into Pinecone
        index.upsert(vectors=vectors)

print("PDF files have been successfully indexed with embeddings and metadata.")