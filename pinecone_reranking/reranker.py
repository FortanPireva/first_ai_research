import pinecone
from sentence_transformers import SentenceTransformer

# Initialize Pinecone
pinecone.init(api_key='YOUR_PINECONE_API_KEY', environment='us-west1-gcp')
index_name = "pdf-index"
index = pinecone.Index(index_name)

# Load the embedding model for the initial retrieval
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the query
query = "What is the battery life of the device?"

# Generate query embedding using the sentence transformer
query_embedding = embed_model.encode(query).tolist()

# Query Pinecone to get top 5 similar chunks
result = index.query(queries=[query_embedding], top_k=5, include_metadata=True)

# Extract the text chunks and metadata from the results
retrieved_texts = [match['metadata']['chunk_text'] for match in result['matches']]

# Print retrieved results (before reranking)
print("Retrieved Results (Before Reranking):")
for i, text in enumerate(retrieved_texts):
    print(f"Document {i+1}: {text[:200]}...")  # Displaying only the first 200 characters of each chunk

# Update the reranking function to use Pinecone's Rerank Inference API
def rerank(query, docs):
    """Rerank the retrieved documents using Pinecone's Rerank Inference API."""
    rerank_results = index.rerank(
        query=query,
        docs=docs,
        top_k=len(docs)  # Rerank all retrieved documents
    )
    
    # Sort documents by their relevance scores
    ranked_docs = sorted(rerank_results, key=lambda x: x['score'], reverse=True)
    return [(doc['text'], doc['score']) for doc in ranked_docs]

# Apply reranking on the retrieved documents
ranked_results = rerank(query, retrieved_texts)

# Display the ranked results
print("Ranked Results (After Reranking):")
for i, (doc, score) in enumerate(ranked_results):
    print(f"Rank {i+1}: (Score: {score})\n{doc[:200]}...")  # Displaying first 200 characters


