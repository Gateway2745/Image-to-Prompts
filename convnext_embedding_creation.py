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

    for j in range(1, 51):
        image_paths = sorted(glob.glob("./image_hierarchy/part-" + str(j) + "/*.png"))

        image_embeddings = []

        n_images = len(image_paths)
        for i in tqdm(range(n_images)):
            image = Image.open(image_paths[i])            
            image_batch = image_processor(image, return_tensors="pt")['pixel_values'].to(device)
            with torch.no_grad():
                image_features = model(image_batch).pooler_output.cpu().numpy()
                image_embeddings.append(image_features)
            del image_batch

        paths = [p.split("/")[-1] for p in image_paths]
        embedding_dict = dict(zip(paths, image_embeddings))
        np.save("convnext-embeddings/convnext-part-" + str(j)+ ".npy", embedding_dict)
    
if __name__ == "__main__":
    create_convnext_embeddings()

    filenames = glob.glob("./convnext-embeddings/*.npy")

    dfs = {}
    for i, filename in enumerate(filenames):
        d = np.load(filename, allow_pickle=True).item()
        dfs.update(d)

    sorted_dict = dict(sorted(dfs.items()))
    image_embeddings = np.concatenate(list(sorted_dict.values()), axis=0)
    np.save('50k_convnext_embeddings.npy', image_embeddings)