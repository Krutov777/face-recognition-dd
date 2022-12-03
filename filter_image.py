from PIL import ImageEnhance, Image
import cv2


def filter_brightness_for_image(open_cv_image, param_enchance):
    img_pil = Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))
    converter = ImageEnhance.Brightness(img_pil)
    img = converter.enhance(param_enchance)
    return img


def filter_image(np_array_image):
    '''np_array_image - numpy.dnarray'''
    # for color images
    #equ = cv2.equalizeHist(cv2.cvtColor(np_array_image, cv2.COLOR_BGR2GRAY))
    equ = cv2.equalizeHist(np_array_image)
    im_pil = Image.fromarray(cv2.cvtColor(equ, cv2.COLOR_BGR2RGB))
    return im_pil


def filter_image2(np_array_image):
    # for color images
    #equ = cv2.equalizeHist(cv2.cvtColor(np_array_image, cv2.COLOR_BGR2GRAY))
    equ = cv2.equalizeHist(np_array_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(equ)
    im_pil = Image.fromarray(cv2.cvtColor(cl1, cv2.COLOR_BGR2RGB))
    return im_pil
