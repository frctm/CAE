from glob import glob
import numpy as np
from PIL import Image


class DataLoader:
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def get_data(self):
        file_path_list = glob(self.dir_path + "*.pbm")

        data = []
        code_list = []
        for file_path in file_path_list:
            code = file_path.split("\\")[-1][:-4]
            image = Image.open(file_path)
            array = np.asarray(image).astype(np.int32)
            data.append(array)
            code_list.append(int(code))
        return np.array(data), code_list


if __name__ == "__main__":
    data_loader = DataLoader(".\\src\\data\\char_image\\")

    data, code_list = data_loader.get_data()
