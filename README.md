# BraTS2023 GLI Segmentation

Clean BraTS2023 GLI experiment pipeline for 2D U-Net baseline training and 3D volume evaluation.

## Run Exp2023_001 on Kaggle

1. Create a Kaggle notebook with GPU enabled.
2. Add the dataset:
   `luumsk/asnr-miccai-brats-2023-gli-challenge-training-data`
3. Clone this repo:

```bash
git clone https://github.com/DucPh4t/brats2023.git
cd brats2023
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Train Exp2023_001:

```bash
python main.py --config configs/exp2023_001.yaml --mode train
```

If Kaggle is close to timing out, stop at a chosen epoch:

```bash
python main.py --config configs/exp2023_001.yaml --mode train --stop_epoch 10
```

Resume later:

```bash
python main.py \
  --config configs/exp2023_001.yaml \
  --mode train \
  --resume_path outputs/exp2023_001/last_checkpoint.pth
```

6. Evaluate the best checkpoint on the test split:

```bash
python main.py --config configs/exp2023_001.yaml --mode eval
```

Outputs are written to `outputs/exp2023_001/`.

## Exp2023_001 Setup

- Split: sequential 80/10/10
- Normalization: brain-masked min-max
- Sampling: fixed all 155 slices
- Model: 2D U-Net, 32 initial features, 4 input modalities, 3 region outputs
- Loss: Dice
- Optimizer: Adam
- Augmentation: disabled
