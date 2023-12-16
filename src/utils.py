import os
import json
import numpy as np

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def read_all_json(path):
    data = []
    for file in os.listdir(path):
        if file.endswith('.json'):
            data.append(read_json(os.path.join(path, file)))
    return [json.loads(d[0]) for d in data]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))