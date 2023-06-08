# Image-to-Prompts

## CSE 252D Advanced Computer Vision Mini Project
### Team Members: Sai Sree Harsha, Rohit Ramaprasad, Omkar Bhope, Yejin Jeon

Text-to-image generative models represent a powerful and innovative approach for creating visual artwork. The rising popularity of these models has given rise to the new field of prompt engineering. While there has been significant progress in prompt engineering for text generation purposes, less work has been done to rigorously examine how users can prompt generative frameworks with natural language for visual generation purposes. In this project, we propose a novel Transformer based ensemble model for the task of predicting the text prompt given a generated image. The predicted text prompt can then be edited and used to generate new images similar to the existing one. Our proposed ensemble model uses embeddings derived from several models such as ConvNext, CLIP and BLIP, and leverages the attention mechanism to fuse these embeddings using a transformer encoder model. We train and evaluate our proposed transformer ensemble model using a large dataset of (prompt, image) pairs from DiffusionDB, and show that our model is able to generate text prompts similar to the prompts used to generate the image.

![alt text](./acv_ensemble.png)

## Demo:

The Jupyter notebook demonstrating our proposed model pipeline including embedding generation and inference of our Transformer ensemble model can be found at ![demo_notebook.ipynb](https://github.com/Gateway2745/Image-to-Prompts/blob/main/demo_notebook.ipynb)

## Requirements:
```
!pip install sentence_transformers
!pip install ftfy regex tqdm
!pip install git+https://github.com/openai/CLIP.git
!pip install pytorch_lightning torchmetrics
!pip install clip-interrogator==0.6.0
```

## Data:

## Embedding Generation:

## Training:
```
python train.py config.yml
```

## Evaluation:

Best checkpoint for 50k - [Drive link to ckpt](https://drive.google.com/file/d/17l8Fsh2VTwJH0HrprU3GtjFtd8maSZ37/view?usp=share_link)

Best checkpoint for 10k - [Drive link to ckpt](https://drive.google.com/file/d/1AXGAxEMAdEC7Cb8IXL_3ngLcmujx-V00/view?usp=share_link)
## Demo:
