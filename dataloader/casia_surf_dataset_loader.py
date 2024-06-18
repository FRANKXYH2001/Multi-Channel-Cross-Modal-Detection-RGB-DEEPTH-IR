import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from random import shuffle

class CASIASURFDataset(Dataset):
    """
    Custom dataloader for the CASIA-SURF dataset.
    """
    def __init__(self, data_base_dir, csv_file_path, preprocess_transform, custom_transform, is_train=True):
        """
        Parameters
        ----------
        data_base_dir: str
            Base directory where data is located. Base dir + relative path in csv should give the sample paths.
        csv_file_path: str
            Path to csv file to read sample data from.
        preprocess_transform: Torch transformer
            Transformation to be applied as preprocessing to the input images.
        custom_transform: function
            Custom wrapper function.
        is_train: bool
            Flag indicating whether the dataset is used for training or not.
        """
        self.data_base_dir = data_base_dir
        self.is_train = is_train
        self.preprocess_transform = preprocess_transform
        self.custom_transform = custom_transform

        self.img_data, self.image_types = self._get_samples_from_csv(csv_file_path)

        """
        For RGB images, normalization was performed using the mean and 
        standard deviation of the ImageNet dataset.
        For Depth and IR images, pixel values were adjusted from a range of [0, 1] 
        to centered at 0 with a range of [-1, 1] using a normalization method 
        with a mean and standard deviation of 0.5.
        """
        self.tensor_norms = {
            'rgb_norm': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'depth_norm': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]),
            'ir_norm': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        }

    def _get_samples_from_csv(self, csv_file_path):
        samples = []
        image_types = []
        with open(csv_file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            headers = next(csv_reader)  # Read the header
            image_types = headers[:2]  # The first two columns are image types
            for row in csv_reader:
                img1_path = os.path.join(self.data_base_dir, row[0])
                img2_path = os.path.join(self.data_base_dir, row[1])
                label = int(row[2])
                if not (os.path.isfile(img1_path) and os.path.isfile(img2_path)):
                    continue  # Skip if any file does not exist
                sample = CASIASURFSample(img1_path, img2_path, label, image_types)
                samples.append(sample)
        return samples, image_types

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        datasample = self.img_data[idx]
        img1, img2 = datasample.load_images()

        img1 = self.preprocess_transform(img1)
        img2 = self.preprocess_transform(img2)

        if 'Rgb_Path' in self.image_types[0]:
            img1_tensor = self.tensor_norms['rgb_norm'](img1)
            img1_type = 'rgb'
        elif 'Depth_Path' in self.image_types[0]:
            img1_tensor = self.tensor_norms['depth_norm'](img1)
            img1_type = 'deepth'
        elif 'IR_Path' in self.image_types[0]:
            img1_tensor = self.tensor_norms['ir_norm'](img1)
            img1_type = 'ir'

        if 'Rgb_Path' in self.image_types[1]:
            img2_tensor = self.tensor_norms['rgb_norm'](img2)
            img2_type = 'rgb'
        elif 'Depth_Path' in self.image_types[1]:
            img2_tensor = self.tensor_norms['depth_norm'](img2)
            img2_type = 'deepth'
        elif 'IR_Path' in self.image_types[1]:
            img2_tensor = self.tensor_norms['ir_norm'](img2)
            img2_type = 'ir'

        label = datasample.label
        return self.custom_transform(img1_tensor, img1_type, img2_tensor, img2_type, label, datasample.id)

class CASIASURFSample:
    def __init__(self, img1_path, img2_path, label, image_types):
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.label = label
        self.image_types = image_types

    def load_images(self):
        img1 = Image.open(self.img1_path).convert('RGB' if 'Rgb_Path' in self.image_types[0] else 'L')
        img2 = Image.open(self.img2_path).convert('RGB' if 'Rgb_Path' in self.image_types[1] else 'L')
        return img1, img2
