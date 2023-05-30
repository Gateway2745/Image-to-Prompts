import torch
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, ConvNextModel

def create_convnext_embeddings():
    device = "cuda"
    image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
    model = ConvNextModel.from_pretrained("facebook/convnext-tiny-224").to(device=device)

    image_paths = sorted(glob.glob("./images/*.png"))

    image_embeddings = []

    n_images = len(image_paths)
    for batch_idx in tqdm(range(0, n_images, 50)):
        batch_images = []
        for i in range(batch_idx, batch_idx+50):
            image = image_processor(Image.open(image_paths[i]), return_tensors="pt")['pixel_values']
            batch_images.append(image)
    
    image_batch = torch.cat(batch_images, dim=0).to(device)
    with torch.no_grad():
        image_features = model(image_batch).pooler_output.cpu().numpy()
        image_embeddings.append(image_features)
        
        
    image_embeddings = np.concatenate(image_embeddings, axis=0)

    np.save('img_embeddings_convnext.npy', image_embeddings)
    
if __name__ == "__main__":
    create_convnext_embeddings()