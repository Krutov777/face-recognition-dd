import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

mtcnn = MTCNN(image_size=160, margin=0)
model = InceptionResnetV1(pretrained='vggface2').eval()
model.classify = True

def set_model(a = 0):
    if a == 0:
        model = InceptionResnetV1(pretrained='vggface2').eval()
    else:
        model = InceptionResnetV1(pretrained='casia-webface').eval()

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def compare_images(img1, img2):
    
    try:
        img1_croped = mtcnn(img1)#torch.from_numpy(np.array(img1))
        img2_croped = mtcnn(img2)#torch.from_numpy(np.array(img2))
    
        img1_probs = model(img1_croped.unsqueeze(0)).detach().numpy()[0]
        img2_probs = model(img2_croped.unsqueeze(0)).detach().numpy()[0]
        sim = cosin_metric(img1_probs, img2_probs)
    except:
        sim = 0.0

    return sim

def test_compare(img1_path, img2_path):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    img1_croped = mtcnn(img1)
    img2_croped = mtcnn(img2)

    img1_probs = model(img1_croped.unsqueeze(0)).detach().numpy()[0]
    img2_probs = model(img2_croped.unsqueeze(0)).detach().numpy()[0]
    sim = cosin_metric(img1_probs, img2_probs)
    print(sim)
    return sim