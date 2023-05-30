import torch
import clip
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image

def create_clip_embeddings():
    model, preprocess = clip.load("ViT-B/32", device="cuda")

    image_paths = sorted(glob.glob("./images/*.png"))

    image_embeddings = []

    n_images = len(image_paths)
    for batch_idx in tqdm(range(0, n_images, 50)):
        batch_images = []
        for i in range(batch_idx, batch_idx+50):
            image = preprocess(Image.open(image_paths[i])).unsqueeze(0)
            batch_images.append(image)
    
    image_batch = torch.cat(batch_images, dim=0).to("cuda")
    with torch.no_grad():
        image_features = model.encode_image(image_batch).cpu().numpy()
        image_embeddings.append(image_features)
        
    image_embeddings = np.concatenate(image_embeddings, axis=0)

    np.save('img_embeddings_clip_vit32.npy', image_embeddings)
    
if __name__ == "__main__":
    create_clip_embeddings()