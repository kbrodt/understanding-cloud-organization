#!/bin/bash


ENCODER=efficientnet-b7
WORK_DIR=effnet7_scse_test
EXP=best_tta3
GPU=0


CUDA_VISIBLE_DEVICES=$GPU python -m src.train --work-dir $WORK_DIR --encoder $ENCODER --cls --n-folds 5 --fold 0

CUDA_VISIBLE_DEVICES=$GPU python -m src.train --work-dir $WORK_DIR --encoder $ENCODER --cls --n-folds 5 --fold 1

CUDA_VISIBLE_DEVICES=$GPU python -m src.train --work-dir $WORK_DIR --encoder $ENCODER --cls --n-folds 5 --fold 2

CUDA_VISIBLE_DEVICES=$GPU python -m src.train --work-dir $WORK_DIR --encoder $ENCODER --cls --n-folds 5 --fold 3

CUDA_VISIBLE_DEVICES=$GPU python -m src.train --work-dir $WORK_DIR --encoder $ENCODER --cls --n-folds 5 --fold 4


CUDA_VISIBLE_DEVICES=$GPU python -m src.predict --load ./${WORK_DIR}/${ENCODER}_b12_adam_lr0.001_c1_fold0/best.pth --tta 3 --save ./${WORK_DIR}/${EXP}

CUDA_VISIBLE_DEVICES=$GPU python -m src.predict --load ./${WORK_DIR}/${ENCODER}_b12_adam_lr0.001_c1_fold1/best.pth --tta 3 --save ./${WORK_DIR}/${EXP}

CUDA_VISIBLE_DEVICES=$GPU python -m src.predict --load ./${WORK_DIR}/${ENCODER}_b12_adam_lr0.001_c1_fold2/best.pth --tta 3 --save ./${WORK_DIR}/${EXP}

CUDA_VISIBLE_DEVICES=$GPU python -m src.predict --load ./${WORK_DIR}/${ENCODER}_b12_adam_lr0.001_c1_fold3/best.pth --tta 3 --save ./${WORK_DIR}/${EXP}

CUDA_VISIBLE_DEVICES=$GPU python -m src.predict --load ./${WORK_DIR}/${ENCODER}_b12_adam_lr0.001_c1_fold4/best.pth --tta 3 --save ./${WORK_DIR}/${EXP}

CUDA_VISIBLE_DEVICES=$GPU python -m src.thresh_search --exp ./${WORK_DIR}/${EXP}

CUDA_VISIBLE_DEVICES=$GPU python -m src.submit --exp ./${WORK_DIR}