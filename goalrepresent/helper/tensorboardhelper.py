import numpy as np
import torch.nn.functional as F

def resize_embeddings(embedding_images, sprite_size=8192):
    image_size = max(embedding_images.shape[2], embedding_images.shape[3])
    n_images = np.ceil(np.sqrt(len(embedding_images)))
    if n_images * image_size <= sprite_size:
        return embedding_images
    else:
        image_ratio = sprite_size / (n_images * image_size)
        return F.interpolate(embedding_images, size=int(image_size*image_ratio))