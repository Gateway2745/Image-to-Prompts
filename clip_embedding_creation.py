import torch
import clip
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image

def create_clip_embeddings():
    device = "cuda"
    model, preprocess = clip.load("ViT-B/32", device=device)

    for j in range(1, 51):
        image_paths = sorted(glob.glob("./image_hierarchy/part-" + str(j) + "/*.png"))
        image_embeddings = []

        n_images = len(image_paths)
        for i in tqdm(range(n_images)):            
            image_batch = preprocess(Image.open(image_paths[i])).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_batch).cpu().numpy()
                image_embeddings.append(image_features)
            del image_batch
        
        paths = [p.split("/")[-1] for p in image_paths]
        embedding_dict = dict(zip(paths, image_embeddings))
        np.save("clip-embeddings/clip-part-" + str(j)+ ".npy", embedding_dict)
    
if __name__ == "__main__":
    create_clip_embeddings()

    filenames = glob.glob("./clip-embeddings/*.npy")

    dfs = {}
    for i, filename in enumerate(filenames):
        d = np.load(filename, allow_pickle=True).item()
        dfs.update(d)

    sorted_dict = dict(sorted(dfs.items()))
    image_embeddings = np.concatenate(list(sorted_dict.values()), axis=0)
    np.save('50k_clip_embeddings.npy', image_embeddings)