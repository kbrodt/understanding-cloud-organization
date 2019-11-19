# Understanding Clouds from Satellite Images

This is an another one approach to solve the competition from kaggle
[Understanding Clouds from Satellite Images](https://www.kaggle.com/c/understanding_cloud_organization).
The pipeline is almost the same as in [severstal](https://github.com/kbrodt/severstal).

40th place out of 1556 (silver medal) with 0.66348 dice score (top 1 -- 0.67175).

### Prerequisites

- GPU with >27 Gb RAM (e.g. Tesla V100)
- [NVidia apex](https://github.com/NVIDIA/apex)

```bash
pip install -r requirements.txt
```

### Usage

First download the train and test data from the competition link.

To train the model run

```bash
bash ./train.sh
```

This will generates trained models and submission file.
