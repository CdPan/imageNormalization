import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import os
import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
rmb_label = {"其他地": 0, "工业民用地": 1, "林地": 2, "水域": 3, "耕地": 4, "草地":5}


class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = {"其他地": 0, "工业民用地": 1, "林地": 2, "水域": 3, "耕地": 4, "草地":5}
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.tif'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = rmb_label[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info








train_dir = os.path.join('../Test9_efficientNet/fid15_80', "test")

train_transform = transforms.Compose([
    transforms.Resize((80, 80)), # 可以改成你图片近似大小或者模型要求大小
    transforms.ToTensor(),
])

train_data = MyDataset(data_dir=train_dir, transform=train_transform)
# train_loader = DataLoader(dataset=train_data, batch_size=1000, shuffle=True) # # 图片的mean std
train_loader = DataLoader(dataset=train_data,batch_size=10000, shuffle=True) # # 图片的mean std
train = iter(train_loader).next()[0]  # 图片的mean std

train_mean = np.mean(train.numpy(), axis=(0, 2, 3), keepdims=False)
train_std = np.std(train.numpy(), axis=(0, 2, 3), keepdims=False)


print(train_mean, train_std)


