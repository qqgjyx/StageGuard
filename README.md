# StageGuard

**Backbone-agnostic physiological constraints for neural sleep staging.**

StageGuard improves any neural sleep staging model by enforcing physiological constraints during both training (soft transition penalty) and inference (Semi-Markov constrained decoding). It works across modalities—EEG, actigraphy, cardiorespiratory signals, and bioradar—with modality-specific configurations.

## Key Contributions

1. **Soft Transition Penalty** — A differentiable loss term that discourages physiologically rare stage transitions (e.g., Wake↔REM) during training.
2. **Semi-Markov Decoder** — An augmented Viterbi decoder that enforces minimum bout durations, penalizes rare transitions, and suppresses flip-flop artifacts at inference time.
3. **Signal Quality Integration** — Per-epoch signal quality scores modulate decoder confidence, gracefully degrading predictions for noisy segments.

## Installation

```bash
# With conda
conda env create -f environment.yml
conda activate stageguard

# Or with pip
pip install -e .
```

## Quick Start

```python
import torch
from stageguard import StageGuardWrapper, ModalityConfig
from stageguard.backbones import get_backbone

# Load modality config
config = ModalityConfig.from_yaml("configs/mouse_eeg.yaml")

# Create backbone + StageGuard wrapper
backbone = get_backbone("accusleep", num_classes=config.num_classes)
model = StageGuardWrapper(backbone, config)

# Training step
x = torch.randn(4, 50, 1, 128)  # (batch, epochs, channels, samples)
targets = torch.randint(0, 3, (4, 50))
loss, details = model.training_step(x, targets)
loss.backward()

# Inference with constrained decoding
predictions = model.predict(x)  # (4, 50) numpy array
```

## Wrap Your Own Backbone

Any `nn.Module` that outputs `(B, T, num_classes)` logits works:

```python
import torch.nn as nn

class MyBackbone(nn.Module):
    def forward(self, x):
        # Your architecture here
        return logits  # (B, T, C)

model = StageGuardWrapper(MyBackbone(), config)
```

## Repository Structure

```
StageGuard/
├── configs/               # Modality-specific YAML configurations
│   ├── mouse_eeg.yaml     # AccuSleep: 3-state, 4s epochs
│   ├── actigraphy.yaml    # Sleep-Accel: 2-state, 30s epochs
│   ├── cardiorespiratory.yaml  # SHHS: 3-state, 30s epochs
│   └── bioradar.yaml      # SLEEPBRL: 3-state, 30s epochs
├── stageguard/
│   ├── config.py           # ModalityConfig dataclass + YAML loader
│   ├── losses.py           # SoftTransitionPenalty + stageguard_loss
│   ├── decoder.py          # SemiMarkovDecoder (augmented Viterbi)
│   ├── wrapper.py          # StageGuardWrapper (backbone + loss + decoder)
│   ├── metrics.py          # TVR, FI, accuracy, kappa, F1, sleep architecture
│   ├── sqi.py              # Signal quality index per modality
│   ├── backbones/          # Backbone implementations + registry
│   │   ├── base.py         # BackboneBase ABC
│   │   ├── accusleep.py    # 2-layer CNN
│   │   └── usleep.py       # U-Net encoder-decoder
│   └── data/               # Dataset loaders (expect pre-downloaded data)
│       ├── base.py         # BaseSleepDataset ABC
│       ├── mouse_eeg.py    # AccuSleep mouse EEG/EMG
│       ├── actigraphy.py   # Sleep-Accel wrist actigraphy
│       ├── cardiorespiratory.py  # SHHS cardiorespiratory
│       └── bioradar.py     # SLEEPBRL bioradar
├── tests/                  # Comprehensive test suite
└── examples/
    └── demo.py             # End-to-end synthetic data demo
```

## Datasets

| Dataset | Modality | Classes | Epoch | URL |
|---------|----------|---------|-------|-----|
| AccuSleep | Mouse EEG/EMG | 3 (Wake, NREM, REM) | 4s | [Zenodo](https://zenodo.org/records/4079563) |
| Sleep-Accel | Wrist actigraphy | 2 (Wake, Sleep) | 30s | [PhysioNet](https://physionet.org/content/sleep-accel/1.0.0/) |
| SHHS | Cardiorespiratory | 3 (Wake, Light, Deep) | 30s | [NSRR](https://sleepdata.org/datasets/shhs) |
| SLEEPBRL | Bioradar | 3 (Wake, Light, Deep) | 30s | Contact authors |

> **Note:** Datasets must be downloaded separately. Some require data use agreements. Each loader provides download instructions via `Dataset.download_instructions()`.

## Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `lambda_trans` | λ | 1.0 | Transition penalty weight |
| `epsilon` | ε | 5.0 | Rare transition penalty in decoder |
| `gamma` | γ | 2.0 | Anti-flip-flop penalty |
| `k` | k | 5 | Flip-flop lookback window (epochs) |
| `d_min` | d_min | per-stage | Minimum bout duration (epochs) |
| `d_max` | d_max | 30 | Maximum tracked duration (epochs) |

## Running Tests

```bash
pytest               # Run all tests
pytest -x            # Stop on first failure
pytest tests/test_decoder.py  # Single module
```

## Running the Demo

```bash
python examples/demo.py
```

## License

MIT
