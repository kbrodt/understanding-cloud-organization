import argparse
from pathlib import Path
import multiprocessing

import numpy as np
import tqdm
import torch

from src.dataset import CloudsDS, dev_transform, collate_fn
from src.metric import calc_dice
from src.utils import get_data_groups, save


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/home/data/clouds',
                        help='Path to data')
    parser.add_argument('--exp', type=str, default='./effnet7_scse_test/best_tta3/',
                        help='Path to data')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=314159)

    return parser.parse_args()


args = parse_args()

path_to_data = Path(args.data)
args.fold = 0
train_gps, dev_gps = get_data_groups(path_to_data / 'train.csv.zip', args)
train_gps += dev_gps

train_ds = CloudsDS(train_gps, root=path_to_data / 'train', transform=dev_transform)
train_loader = torch.utils.data.DataLoader(train_ds,
                                           batch_size=8,
                                           shuffle=False,
                                           num_workers=4,
                                           collate_fn=collate_fn)

targets = []
with tqdm.tqdm(train_loader) as pbar:
    for _, y in pbar:
        targets.extend(y.numpy())

path_to_exp = Path(args.exp)
preds = [np.load(path_to_exp / 'pred_masks_tta' / (p + '.npy')) for p, _ in tqdm.tqdm(train_gps)]
cls_preds = np.array([np.load(path_to_exp / 'pred_clss_tta' / (p + '.npy')) for p, _ in tqdm.tqdm(train_gps)])
cls_trgs = np.array([(t.sum((-1, -2)) > 0) for t in tqdm.tqdm(targets)])
assert len(preds) == len(targets) == len(cls_preds) == len(cls_trgs)

def calc_dice_for_all(dice_thresh):
    n_classes = 4
    dices = np.zeros((len(preds), n_classes))
    for i, (p, t) in enumerate(zip(preds, targets)):
        for c in range(n_classes):
            dices[i, c] = calc_dice(p[c], t[c], thresh=dice_thresh)

    return dice_thresh, dices

threshs = np.linspace(0.05, 0.95, 19)

with multiprocessing.Pool(len(threshs)) as p:
    with tqdm.tqdm(threshs) as pbar:
        res = list(p.imap_unordered(func=calc_dice_for_all, iterable=pbar))
res = {th: r for th, r in res}

save(res, path_to_exp / 'dices.pkl')

threshs_cls = np.linspace(0.0, 1, 21)
sgm_areas = {th: np.array([(p > th).sum((1, 2)) for p in preds]) for th in tqdm.tqdm(res)}

def step(cls_thresh, area_thresh, dice_thresh):
    dice = np.zeros(4)
    for cc_p, c_p, d, t in zip(cls_preds, sgm_areas[dice_thresh], res[dice_thresh], cls_trgs):
        for c in range(4):
            if cc_p[c] <= cls_thresh:
                if t[c]:
                    dice[c] += 0
                else:
                    dice[c] += 1
                continue

            if c_p[c] <= area_thresh:
                if t[c]:
                    dice[c] += 0
                else:
                    dice[c] += 1
                continue

            dice[c] += d[c]

    return dice / len(cls_preds)

threshs_areas = np.linspace(0, 20_000, 11)
results = {}
with tqdm.tqdm(threshs_cls) as pbar:
    for th_cls in pbar:
        for th_area in threshs_areas:
            for th_dice in threshs:
                asd = step(th_cls, th_area, th_dice)
                for c in range(4):
                    results[th_cls, th_area, th_dice, c] = asd[c]

sum([(list(filter(lambda x: x[0][-1] == c, sorted(results.items(), key=lambda x: -x[1]))))[0][-1] for c in
     range(4)]) / 4

for c in range(4):
    print(list(filter(lambda x: x[0][-1] == c, sorted(results.items(), key=lambda x: -x[1])))[0])
