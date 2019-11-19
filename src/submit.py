import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
import tqdm

from src.dataset import dev_transform
from src.utils import read_img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/home/data/clouds',
                        help='Path to data')
    parser.add_argument('--exp', type=str, default='./effnet7_scse_test',
                        help='Path to data')

    return parser.parse_args()


def kaggle_rle_encode(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 1
    rle[1::2] -= rle[::2]

    return rle.tolist()


class DS(torch.utils.data.Dataset):
    def __init__(self, imgs, root):
        self.imgs = imgs
        self.root = root

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_id = self.imgs[index]
        mask = np.zeros((320, 480, 4), dtype='float32')

        return dev_transform(read_img(self.root / img_id), mask=mask)[0], img_id


def collate(x):
    x, y = list(zip(*x))

    return torch.stack(x), y


def main():
    args = parse_args()
    path_to_data = Path(args.data)
    test_anns = [p.name for p in (path_to_data / 'test').glob('*.jpg')]
    batch_size = 32
    ds = DS(test_anns, path_to_data / 'test')
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        collate_fn=collate,
        pin_memory=True,
    )

    models = [
        torch.jit.load(str(p)).cuda().eval()
        for p in Path(args.exp).rglob('*.pt')
    ]
    n_models = len(models)
    n_augs = n_models * 4
    maping = dict(zip(range(1, 5), ['Fish', 'Flower', 'Gravel', 'Sugar']))

    def get_submit(
            threshs_cls=[0.6000000000000001, 0.5, 0.6000000000000001, 0.5],
            min_areas=[20000.0, 14000.0, 18000.0, 12000.0],
            threshs_dice=[0.39999999999999997, 0.49999999999999994, 0.44999999999999996, 0.44999999999999996],
    ):
        masks = torch.zeros((batch_size, 4, 320, 480), dtype=torch.float32, device='cuda')
        clsss = torch.zeros((batch_size, 4), dtype=torch.float32, device='cuda')
        samples = []
        with torch.no_grad():
            with tqdm.tqdm(loader, mininterval=2) as pbar:
                for img, anns in pbar:
                    bs = img.size(0)
                    img = img.cuda()

                    masks.zero_()
                    clsss.zero_()
                    for model in models:
                        mask, clss = model(img)
                        masks[:bs] += torch.sigmoid(mask)
                        clsss[:bs] += torch.sigmoid(clss)

                        # vertical flip
                        mask, clss = model(torch.flip(img, dims=[-1]))
                        masks[:bs] += torch.flip(torch.sigmoid(mask), dims=[-1])
                        clsss[:bs] += torch.sigmoid(clss)

                        # horizontal flip
                        mask, clss = model(torch.flip(img, dims=[-2]))
                        masks[:bs] += torch.flip(torch.sigmoid(mask), dims=[-2])
                        clsss[:bs] += torch.sigmoid(clss)

                        # vertical + horizontal flip
                        mask, clss = model(torch.flip(img, dims=[-1, -2]))
                        masks[:bs] += torch.flip(torch.sigmoid(mask), dims=[-1, -2])
                        clsss[:bs] += torch.sigmoid(clss)

                    masks /= n_augs
                    clsss /= n_augs
                    for mask, clss, annotation in zip(masks, clsss, anns):
                        for cls, m in enumerate(mask):
                            if clss[cls] <= threshs_cls[cls]:
                                continue

                            m = m > threshs_dice[cls]
                            if m.sum() <= min_areas[cls]:
                                continue

                            m = m.cpu().numpy().astype('float32')
                            m = A.Resize(350, 525)(image=np.zeros((320, 480, 3), dtype='uint8'),
                                                   mask=m)['mask']
                            rle = kaggle_rle_encode(m.astype('bool'))
                            samples.append({
                                'ImageId': annotation,
                                'EncodedPixels': ' '.join(map(str, rle)),
                                'ClassId': maping[cls + 1]
                            })

        submission = pd.DataFrame(samples, columns=['ImageId', 'EncodedPixels', 'ClassId'])
        submission['Image_Label'] = submission['ImageId'] + '_' + submission['ClassId']

        return submission

    sub = get_submit()
    sample_sub = pd.read_csv(path_to_data / 'sample_submission.csv', usecols=['Image_Label'])
    sub = pd.merge(sample_sub, sub, how='left', on='Image_Label')
    sub[['Image_Label', 'EncodedPixels']].to_csv('./submission.csv', index=False)


if __name__ == '__main__':
    main()
