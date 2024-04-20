import openai
from openai import OpenAI
import pinecone  
from pinecone import Pinecone, ServerlessSpec
import os 


#OLD CODE, DON'T USE. I WANNA KEEP IT CUZ IT LOOKS COOL


#Make sure to input your own API key in environment variables on windows 10 & 11
#You can input api keys in linux with the commands: export OPENAI_API_KEY=(apikey)   &     export PINECONE_API_KEY=(apikey)
openai.api_key = os.getenv("OPENAI_API_KEY") 
client = OpenAI(api_key= os.getenv("OPENAI_API_KEY") )


pinecone_client = pinecone.Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-east-1" 
) 

#This will be changed in the future
index_name = 'my-embeddings'

if index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        index_name,
        dimension=1536,  
        metric='cosine',  
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
paragraphlist = data.replace('asdijhgoiarswngriaeo', 'aspifghisafghlofnj').split("ENDLINE")
#didnt work wthout the data.replace statement and i've been using it ever since, might work without it test it

x=0
for text_to_embed in paragraphlist:
    embedding = get_embedding(text_to_embed)
    x = int(x)
    x = x + 1
    x = str(x)
    index.upsert([(x, embedding)]) 
