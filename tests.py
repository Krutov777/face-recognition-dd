import numpy as np
import pickle
import io
from PIL import Image

TEST_DATA_FILE = 'pairs.txt'
TEST_DATA_FILE_CROPED = 'test_pairs.txt'
TEST_DATA_DIR = 'lfw/'

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
    with open(TEST_DATA_FILE_CROPED) as f:
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
    pairs = parse_images()
    image_pairs = []
    for pair in pairs:

        img1 = Image.open(pair.get_first_path())
        img2 = Image.open(pair.get_second_path())

        np_arr_img1 = np.array(img1)
        np_arr_img2 = np.array(img2)
        
        image_pairs.append((np_arr_img1, np_arr_img2))


def run_their_tests():
    infile = open('pairs.pkl', 'rb')
    pairs = pickle.load(infile)
    infile.close()

    for pair in pairs:
        input_img = Image.open(io.BytesIO(pair[0]))
        target_img = Image.open(io.BytesIO(pair[1]))



run_own_tests()