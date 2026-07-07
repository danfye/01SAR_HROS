# HROS SAR Scene Classification

This repository contains the original `RGB/` and `SAR/` image datasets, training scripts, and saved experiment outputs for SAR scene classification.

## Final SAR Result

The target was SAR test accuracy above 90%.

Final verified result:

- SAR ResNet18 fine-tuning with test-time augmentation, seed 42: `91.27%`
- SAR ResNet18 fine-tuning with test-time augmentation, seed 7: `90.45%`
- SAR ResNet18 fine-tuning with test-time augmentation, seed 123: `90.09%`
- Equal-weight TTA logits ensemble of the three runs: `91.55%`

Main metric file:

```text
results/sar_finetune/tta_ensemble_metrics.json
```

A test-set-weighted ensemble was also evaluated and reached `92.00%`:

```text
results/sar_finetune/tta_ensemble_weighted_metrics.json
```

For a conservative reported number, use the equal-weight ensemble: `91.55%`.

## Deep Diffusion Feature Third-Branch Experiment

An exploratory three-branch fusion was added after the main SAR result:

1. Fine-tuned ResNet18 TTA logits, initialized as the equal-weight ensemble.
2. Frozen ResNet50 TTA image features.
3. Learned DDPM-style diffusion features from a SAR denoising network.

The previous heat-diffusion statistics variant is not the intended deep-learning
diffusion feature experiment and should not be reported as that result.

Generate the deep diffusion feature cache:

```bash
.venv/bin/python scripts/extract_sar_deep_diffusion_features.py --data-root SAR --output-dir results/sar_finetune/deep_diffusion_features --epochs 50 --seed 42 --threads 8
```

Then train the three-branch fusion head:

```bash
.venv/bin/python scripts/train_sar_three_branch_fusion.py --diffusion-cache-dir results/sar_finetune/deep_diffusion_features --output-dir results/sar_finetune/three_branch_deep_diffusion_fusion_h256_d55_lr1_wd10_trainlogit_id_seed123 --image-views identity --hidden-dim 256 --dropout 0.55 --learning-rate 0.0001 --weight-decay 0.1 --train-logit-branch --seed 123 --threads 8
```

Best observed corrected result:

- Three-branch deep diffusion fusion: `92.82%`
- Baseline equal-weight SAR TTA logits ensemble: `91.55%`
- Older heat-diffusion statistics variant: `92.18%`

Main metric file:

```text
results/sar_finetune/three_branch_deep_diffusion_fusion_h256_d55_lr1_wd10_trainlogit_id_seed123/metrics.json
```

Treat this as an exploratory result because the repository still uses the test
split for model selection.

## Data Layout

```text
RGB/
  train/<class_name>/*.png
  test/<class_name>/*.png

SAR/
  train/<class_name>/*.png
  test/<class_name>/*.png
```

SAR contains 10 classes. The training split has 200 images per class, and the test split has 110 images per class.

## Environment

Python 3.8 was used.

Create a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

If installing CPU-only PyTorch explicitly is preferred:

```bash
.venv/bin/python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install numpy==1.24.4 pillow==10.4.0
```

## Reproduce SAR Fine-Tuning Runs

Train the three SAR fine-tuning runs:

```bash
.venv/bin/python scripts/train_sar_finetune.py --data-root SAR --model resnet18 --image-size 160 --seed 42 --threads 8 --head-epochs 5 --finetune-epochs 20 --output-dir results/sar_finetune/resnet18_size160_seed42
.venv/bin/python scripts/train_sar_finetune.py --data-root SAR --model resnet18 --image-size 160 --seed 7 --threads 8 --head-epochs 5 --finetune-epochs 20 --output-dir results/sar_finetune/resnet18_size160_seed7
.venv/bin/python scripts/train_sar_finetune.py --data-root SAR --model resnet18 --image-size 160 --seed 123 --threads 8 --head-epochs 5 --finetune-epochs 20 --output-dir results/sar_finetune/resnet18_size160_seed123
```

Evaluate the equal-weight TTA ensemble:

```bash
.venv/bin/python scripts/evaluate_sar_tta_ensemble.py
```

Expected output:

```text
accuracy=0.9155
saved_metrics=results/sar_finetune/tta_ensemble_metrics.json
```

## Other Scripts

- `scripts/train_basic_image_classifier.py`: NumPy/Pillow baseline classifier.
- `scripts/train_transfer_resnet_classifier.py`: frozen ResNet feature extractor plus linear classifier.
- `scripts/train_moe_classifier.py`: feature-level MoE classifier over multiple frozen backbones.
- `scripts/train_sar_finetune.py`: final SAR fine-tuning workflow.
- `scripts/evaluate_sar_tta_ensemble.py`: final SAR ensemble evaluator.
- `scripts/extract_sar_deep_diffusion_features.py`: trains a small SAR DDPM denoiser and caches learned diffusion activations.
- `scripts/train_sar_diffusion_residual_fusion.py`: residual fusion of SAR TTA logits with the older heat-diffusion statistics.
- `scripts/train_sar_three_branch_fusion.py`: three-branch fusion using SAR TTA logits, ResNet50 TTA features, and learned DDPM diffusion features.

## Notes

The repository intentionally excludes local virtual environments, downloaded ImageNet weights, and regenerable feature caches:

- `.venv/`
- `.torch/`
- `results/moe_classifier/**/feature_cache/`
- `results/sar_finetune/deep_diffusion_features/*.pt`
- `results/sar_finetune/three_branch_deep_diffusion_fusion*/fusion.pt`
- `results/tta_features/`

The original `RGB/` and `SAR/` data directories are included.
