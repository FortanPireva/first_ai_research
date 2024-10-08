from pinecone import Pinecone
from openai import OpenAI
import os
open_ai_api_key = "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"

# Initialize Pinecone
pc = Pinecone(api_key='6c634c24-f8cb-44a0-829f-00532451a537')
index_name = "polyloop-index"
index = pc.Index(index_name)

# Initialize OpenAI client
client = OpenAI(api_key=open_ai_api_key)

# Function to get embeddings using OpenAI
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Function to get GPT-4 response
def get_gpt4_response(query, context):
    prompt = f"""
    Query: {query}
    
    Context:
    {context}
    
    Based on the above context, please provide a comprehensive answer to the query. If the context doesn't contain enough information to fully answer the query, please state that and provide the best possible answer with the available information.
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides accurate information based on the given context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    
    return response.choices[0].message.content

# Define the query
query = "How save the children aligns with climate projects?"

# Generate query embedding using OpenAI
query_embedding = get_embedding(query)

# Query Pinecone to get top 100 similar chunks
result = index.query(vector=query_embedding, top_k=100, include_metadata=True)

# Extract the text chunks and metadata from the results
retrieved_texts = [match['metadata']['chunk_text'] for match in result['matches']]

# Print retrieved results (before reranking)
print("Retrieved Results (Before Reranking):")
for i, text in enumerate(retrieved_texts[:5]):  # Print only first 5 for brevity
    print(f"Document {i+1}: {text[:200]}...  \n")

# Prepare context for GPT-4 (indexed results)
indexed_context = "\n\n".join(retrieved_texts[:5])  # Use top 5 results as context

# Get GPT-4 response for indexed results
indexed_gpt4_response = get_gpt4_response(query, indexed_context)

# Update the reranking function to use Pinecone's Rerank Inference API
def rerank(query, docs):
    """Rerank the retrieved documents using Pinecone's Rerank Inference API."""
    rerank_result = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=docs,  
        return_documents=True,
        top_n=10  # Rerank all retrieved documents
    )
    print("Rerank Result:", rerank_result)
    
    # Extract the reranked documents and their scores
    ranked_docs = [(item['document']['text'], item['score']) for item in rerank_result.data]
    
    # Sort documents by their relevance scores (highest to lowest)
    ranked_docs.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_docs

# Apply reranking on the retrieved documents
ranked_results = rerank(query, retrieved_texts)

# Display the ranked results
print("Ranked Results (After Reranking):")
for i, (doc, score) in enumerate(ranked_results[:5]):  # Print only first 5 for brevity
    print(f"Rank {i+1}: (Score: {score})\n{doc[:200]} \n")

# Prepare context for GPT-4 (reranked results)
reranked_context = "\n\n".join([doc for doc, _ in ranked_results[:5]])  # Use top 5 results as context

# Get GPT-4 response for reranked results
reranked_gpt4_response = get_gpt4_response(query, reranked_context)

print("\nGPT-4 Responses:")
print("=" * 50)
print("Response based on indexed (not reranked) results:")
print(indexed_gpt4_response)
print("\n" + "=" * 50)
print("Response based on reranked results:")
print(reranked_gpt4_response)


