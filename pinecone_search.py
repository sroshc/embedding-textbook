from openai import OpenAI
import openai
import pinecone  
import os
from pinecone import Pinecone, ServerlessSpec

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

def get_paragraph(id):
    chapter, section, paragraph = id.split(".")
    file_path = os.path.join("textbookfiles", str(chapter), f"{chapter}.{str(section)}.txt")
    textfile = open(file_path, "r", encoding='utf-8') 
    data = textfile.read()
    paragraphlist = data.replace('\n', ' ').split("ENDLINE")
    return paragraphlist[int(paragraph) - 1]


    

user_prompt = input("Enter your search query: ")
results = find_most_similar(user_prompt)

most_similar_id = results.matches[0].id  
similarity_score = results.matches[0].score

#chapter, section, paragraph = most_similar_id.split(".")
#file_path = os.path.join("textbookfiles", str(chapter), f"{chapter}.{str(section)}.txt")
#textfile = open(file_path, "r", encoding='utf-8') 
#data = textfile.read()
#paragraphlist = data.replace('\n', ' ').split("ENDLINE")

print("Most similar embedding ID:", most_similar_id)
print("Similarity score:", similarity_score)
print("Most similar paragraph:", get_paragraph(most_similar_id))
