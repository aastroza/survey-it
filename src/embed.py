from openai import OpenAI

from src.config import EMBEDDING_MODEL

client = OpenAI()

def get_embedding(text, model=EMBEDDING_MODEL):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding