import glob
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, models



if __name__ == "__main__":

    part_captions = glob.glob("./desc*.csv")

    dfs = []
    for i, filename in enumerate(part_captions):
        dfs.append(pd.read_csv(filename))

    all_captions = pd.concat(dfs, ignore_index=True).sort_values('image')
    all_captions.to_csv("all-blip-captions.csv")

    st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    caption_embeddings = st_model.encode(all_captions.prompt.values, batch_size=32, convert_to_numpy=True, device='cuda')

    np.save('50k_clip_intgtr_embeddings.npy', caption_embeddings)