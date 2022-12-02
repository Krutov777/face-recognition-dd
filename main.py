from filter_image import *


if __name__ == '__main__':
    img = Image.open('images\\twister_light1.jpg')
    img.show()
    np_array_img = np.array(img)

    # img2 = filter_brightness_for_image(np_array_img, 1.25)
    # img2.show()

    np_array_img2 = np.array(np_array_img)
    img2_after_filter_brightness = filter_image(np_array_img2)
    img2_after_filter_brightness.show()

    img2_after_filter2_brightness = filter_image2(np_array_img2)
    img2_after_filter2_brightness.show()
