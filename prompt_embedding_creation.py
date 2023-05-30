import glob
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def create_prompt_embeddings():
    st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    prompt_tuples = []

    for path in sorted(glob.glob("./images/*.json")):
        with open(path, "r") as f:
            prompts = json.load(f)
            prompt_tuples += [(k,v['p']) for k,v in prompts.items()]

    prompt_tuples.sort()
    prompt_strings = [x[1] for x in prompt_tuples]

    prompt_embeddings = st_model.encode(prompt_strings, batch_size=32, convert_to_numpy=True, device='cuda')

    np.save('prompt_embeddings.npy', prompt_embeddings)
    
if __name__ == "__main__":
    create_prompt_embeddings()