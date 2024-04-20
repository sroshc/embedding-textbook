from openai import OpenAI
import openai
import pinecone  
import os
from pinecone import Pinecone, ServerlessSpec

# Environment and API Key Setup
openai.api_key = os.getenv("OPENAI_API_KEY") 
#client = OpenAI(api_key= os.getenv("OPENAI_API_KEY") )
client = OpenAI(api_key="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")


pinecone_client = pinecone.Pinecone(
    #api_key=os.getenv("PINECONE_API_KEY"),
    api_key="cbded4ea-6f1d-4bc1-a089-9d8a28dae104"
    #environment="us-east-1" 
) 

index = pinecone_client.Index('my-embeddings', "https://my-embeddings-goyu0di.svc.aped-4627-b74a.pinecone.io")  

def get_embedding(text, engine='text-embedding-3-small'):  
    result = client.embeddings.create(input=text, model=engine)
    return result.data[0].embedding 


# Similarity Search
def find_most_similar(query_text, top_k=1):
    query_embedding = get_embedding(query_text, engine='text-embedding-3-small')
    results = index.query(
        namespace='',  
        top_k=top_k,
        vector=query_embedding,
    )
    return results



    

# Example Usage
user_prompt = input("Enter your search query: ")
results = find_most_similar(user_prompt)

most_similar_id = results.matches[0].id  # This will be a number between 1-16
similarity_score = results.matches[0].score

print("Most similar embedding ID:", most_similar_id)
print("Similarity score:", similarity_score)
