import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

from src.utils import retrieve_img_mask


preprocess = A.Compose([
    A.Resize(320, 480),
])

p = 0.5
albu_train = A.Compose([
    A.HorizontalFlip(p=p),
    A.VerticalFlip(p=p),
    A.OneOf([
        A.RandomBrightnessContrast(p=1),
        A.RandomGamma(p=1),
    ], p=p),
    A.OneOf([
        A.GaussianBlur(p=1),
        A.GaussNoise(p=1),
        A.IAAAdditiveGaussianNoise(p=1),
    ], p=p),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1),
        A.GridDistortion(p=1),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1),
    ], p=p),
    A.ShiftScaleRotate(border_mode=0),
    A.Normalize(),
    ToTensorV2(),
])

albu_dev = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])


def train_transform(img, mask):
    data = preprocess(image=img, mask=mask)
    img, mask = data['image'], data['mask']
    
    data = albu_train(image=img, mask=mask)
    img, mask = data['image'], data['mask']

    return img, mask.permute(2, 0, 1)


def dev_transform(img, mask):
    data = preprocess(image=img, mask=mask)
    img, mask = data['image'], data['mask']
    
    data = albu_dev(image=img, mask=mask)
    img, mask = data['image'], data['mask']
    
    return img, mask.permute(2, 0, 1)

    
class CloudsDS(torch.utils.data.Dataset):
    def __init__(self, items, root, transform):
        self.items = items
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        img, mask = retrieve_img_mask(item, self.root)
        
        return self.transform(img, mask)


def collate_fn(x):
    x, y = list(zip(*x))

    return torch.stack(x), torch.stack(y)
