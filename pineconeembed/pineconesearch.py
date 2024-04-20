from openai import OpenAI
import openai
import pinecone  
import os
from pinecone import Pinecone, ServerlessSpec

openai.api_key = os.getenv("OPENAI_API_KEY") 
client = OpenAI(api_key= os.getenv("OPENAI_API_KEY") )


pinecone_client = pinecone.Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    #environment="us-east-1" 
) 

index = pinecone_client.Index('my-embeddings', "https://my-embeddings-goyu0di.svc.aped-4627-b74a.pinecone.io")  

def get_embedding(text, engine='text-embedding-3-small'):  
    result = client.embeddings.create(input=text, model=engine)
    return result.data[0].embedding 


def find_most_similar(query_text, top_k=1):
    query_embedding = get_embedding(query_text, engine='text-embedding-3-small')
    results = index.query(
        namespace='',  
        top_k=top_k,
        vector=query_embedding,
    )
    return results



    

user_prompt = input("Enter your search query: ")
results = find_most_similar(user_prompt)

most_similar_id = results.matches[0].id  # still working on making an id system
similarity_score = results.matches[0].score

print("Most similar embedding ID:", most_similar_id)
print("Similarity score:", similarity_score)
