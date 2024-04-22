import glob
import ntpath
import random
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


CELEBA_FORMAT_DATASET = 1
RAFD_FORMAT_DATASET = 2


class GenericDataset(Dataset):
    """Dataset description class"""
    def __init__(self, path, format, selected_attrs, mode, **kwargs):
        self.path = path
        self.format = format
        self.selected_attrs = selected_attrs
        self.mode = mode
        self.transform = kwargs.get("transform", get_default_transforms(mode))

        self.attr2idx = {}
        self.idx2attr = {}
        self.train_dataset = []
        self.test_dataset = []

        if self.format == CELEBA_FORMAT_DATASET:
            self.preprocess_celeba()
        else:
            self.preprocess_rafd()

    def preprocess_celeba(self):
        lines = [line.strip() for line in open(self.path + "/list_attr.txt", "r")]
        cnt_samples, all_attr_names = int(lines[0]), lines[1].split()
        assert set(self.selected_attrs).issubset(set(all_attr_names)), "Invalid selected attrs"
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) <= cnt_samples // 50:
                self.test_dataset.append([self.path + "/images/" + filename, label])
            else:
                self.train_dataset.append([self.path + "/images/" + filename, label])

        print(f'Finished preprocessing CelebA-format dataset at {self.path}: {len(self.train_dataset)} train images, {len(self.test_dataset)} test images')

    def preprocess_rafd(self):
        categs = set(ntpath.basename(categ_folder) for categ_folder in glob.glob(self.path + "/train/*"))
        categs_test = set(ntpath.basename(categ_folder) for categ_folder in glob.glob(self.path + "/test/*"))
        assert categs == categs_test, "Train and test do not have the same categories"
        assert set(self.selected_attrs).issubset(categs), "Invalid selected attrs"
        for i, attr_name in enumerate(categs):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        for subfolder in ["train", "test"]:
            dataset = self.train_dataset if subfolder == "train" else self.test_dataset
            for categ in categs:
                if categ not in self.selected_attrs:
                    continue
                for img_path in glob.glob(self.path + f"/{subfolder}/{categ}/*.jpg"):
                    label = [(attr == categ) for attr in self.selected_attrs]
                    dataset.append([img_path, label])

        print(f'Finished preprocessing RAFD-format dataset at {self.path}: {len(self.train_dataset)} train images, {len(self.test_dataset)} test images')

    def dataset_format(self):
        return self.format

    def label_size(self):
        return len(self.selected_attrs)

    def __len__(self):
        dataset = self.train_dataset if self.mode == "train" else self.test_dataset
        return len(dataset)

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == "train" else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(filename)
        return self.transform(image), torch.FloatTensor(label)


class VisualizationDataset(Dataset):
    def __init__(self, path):
        self.img_paths = []
        for img_path in glob.glob(f'{path}/*.jpg'):
            self.img_paths.append(img_path)

        self.transform = get_default_transforms("test")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        image = Image.open(filename)
        return self.transform(image)


def get_default_transforms(mode):
    transform = []
    if mode == "train":
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(178))
    transform.append(T.Resize(128))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    return transform


def get_loader(dataset, batch_size, mode='train', num_workers=1):
    """Build and return a data loader."""
    data_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=(mode=='train'),
                            num_workers=num_workers)
    return data_loader
