from openai import OpenAI
import openai
import pinecone  
import os
from pinecone import Pinecone, ServerlessSpec

unordered_sentences = []
embedding_sentences = []

try:
    openai.api_key = os.getenv("OPENAI_API_KEY") 
    client = OpenAI(api_key= os.getenv("OPENAI_API_KEY") )
except openai.OpenAIError:
    print("OpenAi API key not found! Manually enter API key!")
    entered_openai_key = input("OpenAi API key: ")
    openai.api_key = entered_openai_key
    client = OpenAI(api_key = entered_openai_key)

try:
    pinecone_client = pinecone.Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment="us-east-1" 
    ) 

except pinecone.exceptions.PineconeConfigurationError:
    print("Pinecone API key not found! Manually enter API key!")
    pinecone_client = pinecone.Pinecone(
        api_key= input("Pinecone API key: "),
        environment="us-east-1" 
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