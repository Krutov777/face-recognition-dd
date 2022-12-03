import numpy as np
import pickle
from PIL import Image
from filter_image import filter_image2
from filter_image import filter_image
from model import compare_images

TEST_DATA_FILE = 'pairs.txt'
TEST_DATA_FILE_CROPED = 'test_pairs.txt'
TEST_DATA_DIR = 'lfw/'
TEST_OUTPUT_FILE = 'output/men'
TEST_INPUT_FILE = 'input/pairs.pkl'


class ImagePair:
    def __init__(self, first_dir, first_img_index, second_dir, second_img_index):
        self.first_dir = first_dir
        self.first_img_index = first_img_index
        self.second_dir = second_dir
        self.second_img_index = second_img_index

    def get_first_path(self):
        return TEST_DATA_DIR + self.first_dir + '/' + self.first_dir + '_' + str(self.first_img_index).zfill(4) + '.jpg'

    def get_second_path(self):
        return TEST_DATA_DIR + self.second_dir + '/' + self.second_dir + '_' + str(self.second_img_index).zfill(4) + '.jpg'


def parse_images():
    pairs = []
    with open(TEST_DATA_FILE) as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            line = line.split('\t')
            if (len(line) == 3):
                pairs.append(
                    ImagePair(
                        line[0],
                        line[1],
                        line[0],
                        line[2],
                    )
                )
            elif (len(line) == 4):
                pairs.append(
                    ImagePair(
                        line[0],
                        line[1],
                        line[2],
                        line[3],
                    )
                )
            else:
                continue
    return pairs


def run_own_tests():
    positive = 0
    all = 0
    pairs = parse_images()
    for pair in pairs:

        img1 = Image.open(pair.get_first_path())
        img2 = Image.open(pair.get_second_path())

        filtered_img1 = filter_image2(np.array(img1))
        filtered_img2 = filter_image2(np.array(img2))

        all += 1
        predict_result = pair.first_dir == pair.second_dir
        result = compare_images(filtered_img1, filtered_img2)

        if predict_result == (result >= 0.8):
            positive += 1

    recall = positive / all
    print(f"\nRECALL = {positive} / {all} = {recall}\n")


def run_their_tests():

    #positive = 0
    #all = 0
    infile = open(TEST_INPUT_FILE, 'rb')
    pairs = pickle.load(infile)
    infile.close()
    probs = []

    for pair in pairs:
        input_img =  filter_image2(pair[0]) #Image.fromarray(np.array(pair[0]))
        target_img = filter_image2(pair[1]) #Image.fromarray(np.array(pair[1]))
        #predict_result = pair[2]
        result = compare_images(input_img, target_img) + 0.08
        if (result > 1.0):
            result = 1.0
        elif (result < 0.0):
            result = 0.0

        #all += 1
        #if bool(predict_result) == (result >= 0.8):
        #    positive += 1

        probs.append(result)
        #print(f"test: {result}")

    #recall = positive / all
    #print(f"\nRECALL = {positive} / {all} = {recall}\n")
    np.save(TEST_OUTPUT_FILE, np.array(probs))
