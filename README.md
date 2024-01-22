# Optiver - Trading at the Close

My code used in the Kaggle competition "[Optiver - Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close)".

## Performance

|    Date    | Milestone | Rank | Score  |
| ---------- | --------- | ---- | ------ |
| 2023-12-20 | Public    | 2828 | 5.3744 |
| 2024-01-16 | Update 1  | 1508 | 4.6535 |

## Techniques

### Modelling

- Predict zero (baseline)
- Predict average (baseline)
- LGBM
  - Final submission
  - 10x less estimators/iterations than most competitors (~100)
  - Lower depth (6-8 levels) than most competitors
- Vision Transformer
  - Each snapshot in time is an image channel
  - Poor performance
- Predict future WAP to reconstruct the target, instead of the target itself
  - Horrible performance

### Data Handling

- Feature engineering based on a public notebook (like everyone else)
  - Removed global features that the model should be able to know
- Reverse engineered the stock index based on public discussion
- For ViT, properly impute missing values

### Online Training

- Retrain model every 7 to 13 days, using newly revealed private data
- Use a rolling window to keep only the latest training data, to avoid timeout
- Add "yesterday's" features as input
  - Halved the rolling window

## Setup

PyTorch + CUDA is very picky when I am using CUDA 11.7.
The default script from official website won't get you anything.
One must identify specific versions that has CUDA 11.7 available.
https://conda.anaconda.org/pytorch/win-64/

For me, the command is:
`conda install pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia`
