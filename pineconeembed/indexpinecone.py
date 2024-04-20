import openai
from openai import OpenAI
import pinecone  
from pinecone import Pinecone, ServerlessSpec
import os 

openai.api_key = os.getenv("OPENAI_API_KEY") 
client = OpenAI(api_key= os.getenv("OPENAI_API_KEY") )


pinecone_client = pinecone.Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-east-1" 
) 

# Create an index on Pinecone (do this outside the loop for efficiency)
index_name = 'my-embeddings'

# Check if the index exists
if index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        index_name,
        dimension=1536,  
        metric='cosine',  # Recommend cosine similarity for embeddings
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ) 
        ) 

index = pinecone_client.Index(index_name)  

def get_embedding(text, engine='text-embedding-3-small'):  
    result = client.embeddings.create(input=text, model=engine)
    return result.data[0].embedding 

    
textfile = open("/workspaces/codespace/gptembed/textbook.txt", "r")
data = textfile.read()
paragraphlist = data.replace('asdijhgoiarswngriaeo', 'aspifghisafghlofnj').split("ENDLINE")\

x=0
for text_to_embed in paragraphlist:
    embedding = get_embedding(text_to_embed)
    x = int(x)
    x = x + 1
    x = str(x)
    # Upsert the embedding to the Pinecone index
    index.upsert([(x, embedding)]) 