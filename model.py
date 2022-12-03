import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

model = InceptionResnetV1(pretrained='vggface2', classify=True).eval().float()
mtcnn = MTCNN(image_size=160)

def set_model(a = 0):
    global model
    if a == 0:
        model = InceptionResnetV1(pretrained='vggface2', classify=True).eval().float()
    else:
        model = InceptionResnetV1(pretrained='casia-webface', classify=True).eval().float()


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def compare_images(img1, img2):
    
    img1_resized = (np.array(img1.resize((160, 160)))).astype(float)/255
    img2_resized = (np.array(img2.resize((160, 160)))).astype(float)/255

    img1_croped = torch.from_numpy(img1_resized)
    img2_croped = torch.from_numpy(img2_resized)

    # shape(3, x, y)
    # shape(x, y, 3)
    if img1_croped.shape[0] != 3:
        img1_croped = img1_croped.permute(2, 0, 1)
    if img2_croped.shape[0] != 3:
        img2_croped = img2_croped.permute(2, 0, 1)

    img1_croped = img1_croped.unsqueeze(0)
    img2_croped = img2_croped.unsqueeze(0)

    img1_probs = model(img1_croped.float()).detach().numpy()[0]
    img2_probs = model(img2_croped.float()).detach().numpy()[0]
    sim = cosin_metric(img1_probs, img2_probs)

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