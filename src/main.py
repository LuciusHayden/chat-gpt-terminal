import subprocess
import requests 
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import ollama
from dotenv import load_dotenv
import os
from openai import OpenAI
import argparse

load_dotenv()
qdrant_api_key = os.getenv("QDRANT_API_KEY")
model = "mistral:7b"

#TODO: unhardcode this url one day
qdrant_client = QdrantClient(
    url="https://0e8a8bf7-1c7b-44ca-ae37-c05ff153c23b.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key=qdrant_api_key
)

openai = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
# openai = OpenAI()

client = QdrantClient("localhost", port=6333)

# client.recreate_collection(
#     collection_name="man_pages",
#     vectors_config=VectorParams(size=768, distance=Distance.COSINE)
# )

def manual(page):
    result = subprocess.run(["man", page], capture_output=True, text=True)
    return result

def embed(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text:v1.5", "prompt": text}
    )
    # response = ollama.embed(model="nomic-embed-text:v1.5", input=text)
    return response.json()['embedding']

def parse_commands(raw: str) -> list[str]:
    commands = []
    for line in raw.splitlines():
        if not "(1)" in line and not "(8)" in line:
            continue
        if not line.strip():
            continue
        parts = line.split(" - ", 1) 
        left = parts[0].strip()
        cmd = left.split(" ", 1)[0]   
        cmd = cmd.split("(")[0].strip()
        commands.append(cmd)
    return list(set(commands))  

def search_database(prompt):
    return client.query_points(
        collection_name="man_pages",
        query=embed(prompt),
        limit=10,
    )

def query_chat_bot(prompt, local=False):
    retrieved_data = search_database(prompt)
    if local:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model":model,
                "stream": False,
                "messages": [
                    {
                      "role": "System",
                      "content": f"Use the following context to find a solution to the problem {retrieved_data}"
                    },
                    {
                      "role": "user",
                      "content": prompt
                    }
                ],
            }
        ).json()
        
    else:
        response = openai.responses.create(
            model="openai/gpt-oss-20b",
            input=f""" Answer the problem as follows, [command needed] ||| [quick reason why] ||| [newline]
                give no addition information: {retrieved_data} \n\n\n problem: {prompt}""",
        )

    return response
    

BATCH_SIZE = 10 
buffer = []

def add_man_pages_to_vector_database():
    man_pages= subprocess.run(["man", "-k", "."], capture_output=True, text=True)
    man_pages = parse_commands(man_pages.stdout)
    i = 1
    for result in man_pages:
        man = manual(result).stdout
        if not man:
            continue 
        chunks = [man[i:i+500] for i in range(0, len(man), 500)]
        for chunk in chunks:
            embedding = embed(chunk)
            buffer.append({
                "id": i,
                "vector": embedding,
                "payload": {"command": result , "text": chunk}
            })
            i+=1
            print((len(buffer)))
            if len(buffer) >= BATCH_SIZE:
                print("clearing buffer")
                client.upsert(
                    collection_name="man_pages",
                    points=buffer
                )
                buffer = []

    if buffer:
        client.upsert(
            collection_name="man_pages",
            points=buffer
        )
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=bool, default=True)
    args = parser.parse_args()
    
    print(args)
    print(query_chat_bot("fix my linux audio"))
    

if __name__ == "__main__":
    main()
        