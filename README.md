# Demystifying the Optimal Fair Classifier in Multi-Class Classification

Official code release for the ICML 2026 paper **"Demystifying the Optimal Fair Classifier in Multi-Class Classification"**.

This repository implements OptFair, a framework for multi-class group-fair classification under demographic parity, equalized opportunity, and equalized odds constraints.

## Method Overview

This paper characterizes the Bayes-optimal fair classifier for multi-class classification and develops in- and post-processing algorithms that provably approach the optimal accuracy-fairness trade-off.

OptFair starts from an analytically tractable formulation of the Bayes-optimal classifier under group-fairness constraints. The formulation expresses fairness constraints through group-specific confusion matrices, introduces an entropic regularizer to obtain a smooth closed-form probabilistic classifier, and converts the resulting dual problem into practical optimization procedures.

The repository contains two main OptFair variants:

- **OptFair-in (`infair`)**: an in-processing method that enforces fairness during training. It reduces the fair-learning problem to a sequence of cost-sensitive classification problems and updates dual variables with a proximal step.
- **OptFair-post (`postfair`)**: a post-processing method that first estimates conditional probabilities from a trained model, then calibrates output probabilities by solving a convex dual optimization problem with plug-in estimation.

The code also includes **ERM (`erm`)** as a standard empirical risk minimization baseline.

## Repository Structure

```text
.
|-- main.py                         # Entry point: parse options, load data, run the selected algorithm
|-- options.py                      # Command-line arguments and optional YAML default loading
|-- config.py                       # Supported algorithms and model names
|-- requirements.txt                # Python dependencies
|-- data/                           # Supported dataset as described in the paper: Adult, ENEM, AcsIncome, CelabA
|-- saved_model/                    # Cached pretrained auxiliary models
|-- log/
|   |-- exp_config/                 # Default experiment configs for OptFair-in/post
|   `-- <dataset>/<algorithm>/      # Runtime logs, configs, and metrics
`-- optimalfair/
    |-- algorithm/
    |   |-- ERM.py                  # ERM baseline
    |   |-- infair.py               # OptFair-in
    |   |-- postfair.py             # OptFair-post
    |   `-- classifierbase.py       # Shared training/evaluation utilities
    `-- utils/
        |-- data_utils.py           # Dataset loading and preprocessing
        |-- model_utils.py          # Fairness metrics, projections, logging utilities
        |-- models.py               # Logistic, MLP, and ResNet models
        `-- latex_utils.py          # Plot formatting helper
```

## Environment Setup

Create a Python environment and install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

PyTorch installation can depend on your CUDA version. If needed, install the matching `torch` and `torchvision` wheels from the official PyTorch instructions, then run the command above for the remaining dependencies.

## Dataset

The dataset loader is implemented in `optimalfair/utils/data_utils.py`. The current code supports:

- **Adult**: included under `data/adult/`; supports `gender`, `race`, and `gender-race` sensitive attributes.
- **CelebA**: expects CelebA files under `data/celeba/raw_data/`, including `list_attr_celeba.txt` and `img_align_celeba/`.
- **ENEM**: expects the ENEM raw file configured in `enem_process`; processed data is cached under `data/enem/processed_data/`.
- **ACSIncome**: loaded through Fairlearn/Folktables-style ACS processing and cached under `data/acsincome/`.
- **COMPAS**: expects `data/compas/raw_data/compas-scores-two-years.csv`.

Set `--generate_data True` to regenerate processed files when the corresponding raw data is available.

## Running OptFair

Run experiments through `main.py`:

```bash
python main.py --algorithm postfair --data adult --sensitive_attr gender --fairness_metric dp --fairness_bound 0.01
```

Common arguments are defined in `options.py`:

- `--algorithm`: algorithm name, usually `erm`, `infair`, or `postfair`.
- `--data`: dataset name, e.g. `adult`, `enem`, `acs`, `celeba`, or `compas`.
- `--sensitive_attr`: sensitive attribute used by the dataset loader.
- `--fairness_metric`: fairness metric, currently `dp` or `eo` in the provided configs; the equal-opportunity path is named `eop` in the algorithm code, while the parser help mentions `eopp`.
- `--fairness_bound`: target fairness constraint level.
- `--model`: model architecture, e.g. `logistic`, `1nn`, `2nn`, or `resnet`.
- `--lr`: learning rate for model training.
- `--num_round`: number of training rounds/epochs for ERM and OptFair-in.
- `--batch_size`: training batch size.
- `--load_param`: whether to load default YAML parameters from `log/exp_config/`.

OptFair shared parameters:

- `--dual_params_lr`: learning rate for dual-variable updates.
- `--dual_params_bound`: L1 radius for the dual-variable constraint.
- `--tau`: entropic regularization strength.

OptFair-in (`infair`) parameters:

- `--inner_round`: number of inner optimization epochs for each dual update.

OptFair-post (`postfair`) parameters:

- `--post_num_round`: number of post-processing optimization rounds.
- `--post_batch_size`: batch size for post-processing; use `0` for full-batch optimization.

When `--load_param` is enabled, defaults are loaded from files such as:

```text
log/exp_config/in-adult-dp.yaml
log/exp_config/post-adult-dp.yaml
```

Command-line arguments can still override the loaded defaults during the final parse.

Example OptFair-in run:

```bash
python main.py --algorithm infair --data adult --sensitive_attr gender --fairness_metric dp --fairness_bound 0.01 --model logistic
```

Example OptFair-post run:

```bash
python main.py --algorithm postfair --data adult --sensitive_attr gender --fairness_metric eo --fairness_bound 0.01 --model logistic
```

Results are written to `log/<dataset>/<algorithm>/<fairness_metric>-<fairness_bound>-<model>-<timestamp>/`, including `config.json` and `metrics.jsonl`.

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{
anonymous2026demystifying,
title={Demystifying the Optimal Fair Classifier in Multi-Class Classification},
author={Anonymous},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=70vztAT0Or}
}
```
