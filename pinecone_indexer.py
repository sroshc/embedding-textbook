import openai
from openai import OpenAI
import pinecone  
from pinecone import Pinecone, ServerlessSpec
import os 

#Make sure to input your own API key in environment variables on windows 10 & 11
#You can input api keys in linux with the commands: export OPENAI_API_KEY=(apikey)   &     export PINECONE_API_KEY=(apikey)
#Input your code from commandline if it doesn't detect it
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

chapter = input("Chapter: ")
section = 1

while True:
    try:
        #textfile = open(f"textbookfiles\{str(chapter)}\13.{str(section)}.txt", "r")
        file_path = os.path.join("textbookfiles", str(chapter), f"13.{str(section)}.txt")
        #textfile = open(file_path, "r")
        textfile = open(file_path, "r", encoding='utf-8') 
    except FileNotFoundError:
        print(f"Chapter {str(chapter)} files have been indexed!")
        break
    data = textfile.read()
    paragraphlist = data.replace('\n', ' ').split("ENDLINE")
    #didnt work wthout the data.replace statement a while ago and i've been using it ever since, might work without, it test it


    paragraph_number = 0
    for text_to_embed in paragraphlist:
        embedding = get_embedding(text_to_embed)

        paragraph_number = paragraph_number + 1     

        index.upsert([(f"{str(chapter)}.{str(section)}.{str(paragraph_number)}", embedding)]) 

    section = section + 1
