from PIL import ImageEnhance, Image
import cv2
import numpy as np
import os
from pathlib import Path


# def covert_image_to_grayscale(image_path):
#     img = Image.open(image_path)
#     return img.convert('L')


def filter_brightness_for_image(open_cv_image, param_enchance):
    img_pil = Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))
    converter = ImageEnhance.Brightness(img_pil)
    img = converter.enhance(param_enchance)
    return img


def filter_image(open_cv_image):
    '''open_cv_image - numpy.dnarray'''
    equ = cv2.equalizeHist(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY))
    im_pil = Image.fromarray(cv2.cvtColor(equ, cv2.COLOR_BGR2RGB))
    return im_pil
    # result_path_to_image = os.path.splitext(
    #     image_path)[0] + '_equ' + Path(image_path).suffix
    # cv2.imwrite(result_path_to_image, equ)


def filter_image2(open_cv_image):
    equ = cv2.equalizeHist(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(equ)
    im_pil = Image.fromarray(cv2.cvtColor(cl1, cv2.COLOR_BGR2RGB))
    return im_pil
