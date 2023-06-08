import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, models

def create_gt_embeddings(path_to_csv, batch_size=1):
  st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
  gt_csv = pd.read_csv(path_to_csv).sort_values('image')
  gt_prompts = gt_csv.prompt.values
  gt_prompt_embeddings = st_model.encode(gt_prompts, batch_size=batch_size, convert_to_numpy=True, device='cuda')
  return gt_prompt_embeddings


if __name__=="__main__":
  gt_prompt_embeddings = create_gt_embeddings("/content/drive/MyDrive/CSE 252D Project/50k_gt_prompts.csv", batch_size=32)
  np.save('50k_gt_prompt_embeddings.npy', gt_prompt_embeddings)
