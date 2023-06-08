import os
import csv
import glob
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm

import os, subprocess
from IPython.display import clear_output, display
from clip_interrogator import Config, Interrogator
from sentence_transformers import SentenceTransformer, models

def image_to_prompt(ci, image, mode):
    ci.config.chunk_size = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    ci.config.flavor_intermediate_count = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image)
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    elif mode == 'fast':
        return ci.interrogate_fast(image)
    elif mode == 'negative':
        return ci.interrogate_negative(image)

def create_clip_intgtr_embeddings(folder_path):
    caption_model_name = 'blip-large' 
    clip_model_name = 'ViT-H-14/laion2b_s32b_b79k'

    config = Config()
    config.clip_model_name = clip_model_name
    config.caption_model_name = caption_model_name
    ci = Interrogator(config)

    prompt_mode = 'fast'
    ci.config.quiet = True


    files = sorted(glob.glob(folder_path))
    prompts = []
    for idx, f in enumerate(tqdm(files)):
        if idx > 0 and idx % 100 == 0:
            clear_output(wait=True)
        image = Image.open(f).convert('RGB')
        prompt = image_to_prompt(ci, image, prompt_mode)
        prompts.append(prompt)

    st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    caption_embeddings = st_model.encode(prompts, batch_size=1, convert_to_numpy=True, device='cuda')
    return caption_embeddings

if __name__ == "__main__":
    caption_embeddings = create_clip_intgtr_embeddings("./images/*.png")
    np.save('50k_clip_intgtr_embeddings.npy', caption_embeddings)
