import torch
import clip
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image

def create_clip_embeddings(folder_path, batch_size):
    model, preprocess = clip.load("ViT-B/32", device="cuda")

    image_paths = sorted(glob.glob(folder_path))

    image_embeddings = []

    n_images = len(image_paths)
    for batch_idx in tqdm(range(0, n_images, batch_size)):
        batch_images = []
        for i in range(batch_idx,min(n_images, batch_idx + batch_size)):
            image = preprocess(Image.open(image_paths[i])).unsqueeze(0)
            batch_images.append(image)
            
        image_batch = torch.cat(batch_images, dim=0).to("cuda")
        with torch.no_grad():
            image_features = model.encode_image(image_batch).cpu().numpy()
            image_embeddings.append(image_features)
        
    image_embeddings = np.concatenate(image_embeddings, axis=0)

    return image_embeddings
    
if __name__ == "__main__":
    image_embeddings = create_clip_embeddings("./images/*.png", batch_size=50)
    np.save('img_embeddings_clip_vit32.npy', image_embeddings)
