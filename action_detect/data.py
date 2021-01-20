import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class PoseDataSet(Dataset):

    def __init__(self, root, is_train = True):
        super(PoseDataSet, self).__init__()
        self.dataset = []

        sub_dir = "train" if is_train else "test"

        for tag in os.listdir(f"{root}/{sub_dir}"):
            file_dir = f"{root}/{sub_dir}/{tag}"
            for img_file in os.listdir(file_dir):
                img_path = f"{file_dir}/{img_file}"
                if tag == 'fall':
                    self.dataset.append((img_path, 0))
                else:
                    self.dataset.append((img_path, 1))
                # print(self.dataset)

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        img = cv2.imread(data[0],cv2.IMREAD_GRAYSCALE) #以灰度图形式读数据
        img = img.reshape(-1)
        img = img/255 #把数据转成[0,1]之间的数据

        tag_one_hot = np.zeros(2)
        tag_one_hot[int(data[1])] = 1

        return np.float32(img),np.float32(tag_one_hot)

if __name__ == '__main__':
    dataset = PoseDataSet('C:/Users/lieweiai/Desktop/human_pose')
    print(dataset[0][1])