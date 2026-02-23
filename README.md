# ENSO Reinforcement Learning (RL)

Multi-year ENSO (El Niño Southern Oscillation) climate control using Proximal Policy Optimization (PPO). This project demonstrates how a reinforcement learning agent can learn control strategies to modulate climate indices and increase the probability of multi-year El Niño or La Niña events.

## 📋 Project Overview

- **Environment**: XRO (Extended Reconstructed Ocean) climate model
- **Algorithm**: PPO (Proximal Policy Optimization) from Stable Baselines 3
- **Goal**: Learn climate control actions that promote multi-year ENSO events
- **Tracking**: Weights & Biases (W&B) integration for experiment monitoring
- **Data**: ORAS5 SST indices from 1979-2022

## 🏗️ Project Structure

```
enso-rl/
├── config/                    # Configuration modules
│   ├── __init__.py
│   ├── env_config.py         # Environment parameters
│   ├── train_config.py       # Training hyperparameters
│   └── wandb_config.py       # W&B settings
│
├── envs/                      # Gymnasium environments
│   ├── __init__.py
│   └── xro_env.py            # XROMultiYearEnv class
│
├── callbacks/                 # Training callbacks
│   ├── __init__.py
│   └── wandb_callback.py     # W&B logging callback
│
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── data_processing.py    # Data loading & preprocessing
│   ├── physics.py            # XRO physics simulation
│   ├── enso_classifier.py    # ENSO event classification
│   └── evaluation.py         # Evaluation utilities
│
├── scripts/                   # Executable scripts
│   ├── __init__.py
│   ├── train.py              # Main training script
│   └── evaluate.py           # Evaluation & analysis script
│
├── notebooks/                 # Jupyter notebooks
│   ├── 01_setup.ipynb        # Initial setup
│   ├── 02_train.ipynb        # Training walkthrough
│   ├── 03_eval.ipynb         # Evaluation analysis
│   └── 04_analysis.ipynb     # Detailed analysis
│
├── data/                      # Data directory
│   └── XRO_indices_oras5.nc   # Observational indices
│
├── models/                    # Saved models (empty)
├── wandb/                     # W&B run logs (generated)
├── enso-rl.ipynb             # Original notebook (preserved)
├── pyproject.toml            # Project dependencies
└── README.md                 # This file
```

## ⚙️ Dependencies

The project uses the following key packages:
- `stable-baselines3`: RL algorithms
- `gymnasium`: Environment interface
- `xarray`: Data handling
- `wandb`: Experiment tracking
- `XRO`: Climate model
- `numpy`, `pandas`, `matplotlib`: Data processing & visualization

See `pyproject.toml` for complete list.

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have Python 3.12+ installed:

```bash
python --version
```

### 2. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using uv (if installed)
uv sync
```

### 3. Data Setup

The observational data is automatically downloaded during training if not present. To pre-download:

```bash
mkdir -p data
wget -c -P data/ https://github.com/senclimate/XRO/raw/main/data/XRO_indices_oras5.nc
```

### 4. Weights & Biases Setup (Optional)

For experiment tracking with W&B:

```bash
# Configure W&B
wandb login

# Update config/wandb_config.py with your entity name if needed
```

## 🎯 Running the Project

### Training the Agent

#### Basic Training (Using Defaults)

```bash
python scripts/train.py
```

#### Training with Custom Hyperparameters

```bash
# Debug mode with reduced training
python scripts/train.py --debug --epochs 50 --lr 0.0001

# Full training with custom learning rate
python scripts/train.py --epochs 6000 --lr 0.0003

# Training without W&B logging
python scripts/train.py --no-wandb
```

#### Available Training Options

```
Options:
  --debug              Enable debug mode with verbose output
  --epochs EPOCHS      Override training epochs (default: 6000)
  --lr LR             Override learning rate (default: 0.0003)
  --no-wandb          Disable Weights & Biases logging
  --help              Show help message
```

### Evaluating the Agent

#### Quick Evaluation

```bash
python scripts/evaluate.py --model ppo_enso_model.zip
```

#### Detailed Analysis

```bash
# All evaluations (basic + intervention + trajectory)
python scripts/evaluate.py --model ppo_enso_model.zip --all

# Basic agent vs baseline comparison
python scripts/evaluate.py --model ppo_enso_model.zip --basic

# Interventional analysis (ablation study showing variable importance)
python scripts/evaluate.py --model ppo_enso_model.zip --intervention --steps 1200

# Trajectory analysis with time series export
python scripts/evaluate.py --model ppo_enso_model.zip --trajectory --months 2400

# Custom evaluation steps
python scripts/evaluate.py --model ppo_enso_model.zip --steps 10000
```

#### Evaluation Outputs

- **Basic Evaluation**: Compares multi-year event frequency (agent vs baseline)
- **Interventional Analysis**: Shows impact of each controllable variable (ablation study)
- **Trajectory Analysis**: Generates `trajectory_analysis.csv` with detailed time series data
- **Results Summary**: Printed to console for immediate analysis

## 📊 Configuration

All configurations are defined in the `config/` directory as Python dataclasses. Modify these files before training to customize behavior.

### Environment Configuration (`config/env_config.py`)

```python
threshold = 1.0                  # ENSO event threshold magnitude
action_scale = [0.15, ...]      # Scaling factors for each control variable
max_steps = None                # None = continuous (no episode resets)
```

### Training Configuration (`config/train_config.py`)

```python
train_epochs = 6000             # Total training timesteps
n_steps = 600                   # Policy update frequency
learning_rate = 0.0003          # PPO learning rate
debug_mode = True               # Verbose training output
```

### W&B Configuration (`config/wandb_config.py`)

```python
project = "enso-rl"
entity = "your-username"        # Change to your W&B username
mode = "online"                 # "online", "offline", or "disabled"
```

## 📈 Training Monitoring

### With Weights & Biases

The training script automatically logs:
- **Real-time metrics**: Episode rewards, policy loss, value loss, KL divergence
- **Hyperparameters**: All settings and configurations
- **Model artifacts**: Trained model checkpoints
- **Evaluation results**: Performance comparison data

View results: https://wandb.ai/your-username/enso-rl

### Local Training Output Example

```
================================
ENSO RL AGENT TRAINING PIPELINE
================================

SETTING UP ENVIRONMENT
======================
1. Loading observational data...
   Loaded 10 variables

2. Preparing XRO model...
   XRO model fitted

3. Initializing Weights & Biases...
   View experiment at: https://wandb.ai/...

4. Creating Gymnasium environment...
   Environment created

STARTING PPO TRAINING
====================
Total timesteps: 6000
Learning rate: 0.0003
...
✓ Training completed successfully!
✓ Model saved to ppo_enso_model.zip
```

## 🔬 Usage Examples

### Example 1: Quick Testing

```bash
# Train for short period to test setup
python scripts/train.py --debug --epochs 100 --lr 0.0001

# Evaluate the test model
python scripts/evaluate.py --model ppo_enso_model.zip --basic
```

### Example 2: Full Training and Analysis Pipeline

```bash
# 1. Train the agent
python scripts/train.py --epochs 6000 --lr 0.0003

# 2. Run comprehensive evaluation
python scripts/evaluate.py --model ppo_enso_model.zip --all

# 3. View results
# - Check console output for statistics summary
# - Check trajectory_analysis.csv for detailed time series data
# - View W&B dashboard at https://wandb.ai/your-username/enso-rl
```

### Example 3: Hyperparameter Test

```bash
# Test different learning rates
for lr in 0.0001 0.0003 0.001; do
    echo "Training with learning rate = $lr"
    python scripts/train.py --epochs 3000 --lr $lr --no-wandb
    python scripts/evaluate.py --model ppo_enso_model.zip --basic
done
```

### Example 4: Disable W&B Logging

```bash
# Train without W&B (useful for offline work or testing)
python scripts/train.py --no-wandb --epochs 100 --debug
```

## 📓 Jupyter Notebooks

Interactive notebooks for detailed exploration:

- **01_setup.ipynb**: Data loading, exploration, and XRO model setup
- **02_train.ipynb**: Step-by-step training walkthrough
- **03_eval.ipynb**: Evaluation and performance analysis
- **04_analysis.ipynb**: Deep dive into ENSO event patterns

Run notebooks:

```bash
# Using Jupyter Lab
jupyter lab

# Using Jupyter Notebook
jupyter notebook

# Or use VSCode integrated notebook support
# Open any .ipynb file in VS Code and select "Run"
```

## 🧪 Understanding Results

### Key Metrics

- **Multi-year Event Probability**: Percentage of time steps containing multi-year ENSO events
- **Delta R (ΔR)**: Change in average reward when disabling each action (more negative = more important)
- **Classified Event**: Type of ENSO event for a trajectory (Multi-year El Niño, La Niña, etc.)
- **Improvement**: How much better the agent performs vs baseline

### Example Output Interpretation

```
With RL Agent Control:      42.5%     # Agent achieves 42.5% multi-year events
Without Control (Baseline): 15.0%     # Baseline achieves 15.0%
Improvement:               +27.5pp   # 27.5 percentage point gain
Improvement ratio:          2.83x    # Agent is 2.83x better than baseline
```

### Interventional Analysis

Lower ΔR values indicate more critical variables:

```
Feature Importance (most important first):
1. IOB     - ΔR: -0.77  (CRITICAL - largest impact)
2. NPMM    - ΔR: -0.64  (CRITICAL - important for multi-year events)
3. WWV     - ΔR: -0.11  (MODERATE - some impact)
4. Others  - ΔR: ~0.00  (MINIMAL - little impact)
```

## 🔧 Troubleshooting

### Model or Data Issues

**Problem**: "Model not found" error
```bash
# Solution: Ensure training completed successfully
ls -la ppo_enso_model.zip
```

**Problem**: Data file not found
```bash
# Solution: Download manually
mkdir -p data
wget -P data/ https://github.com/senclimate/XRO/raw/main/data/XRO_indices_oras5.nc
```

**Problem**: XRO import errors
```bash
# Solution: Reinstall XRO package
pip install --upgrade XRO
```

### W&B Issues

**Problem**: W&B login fails
```bash
# Solution: Re-login or disable W&B
wandb login --relogin
# Or
python scripts/train.py --no-wandb
```

**Problem**: Permission denied on wandb files
```bash
# Solution: Clean wandb cache
rm -rf ~/.wandb
wandb login
```

### Performance Issues

**Problem**: Training is slow or out of memory
```bash
# Solution: Reduce batch size or training steps
python scripts/train.py --epochs 1000 --lr 0.0001 --debug
```

## 📧 Contact & Support

For questions, issues, or suggestions:

- Open a GitHub Issue
- Check existing issues and discussions
- Review the reference materials above

---

**Original Notebook**: `enso-rl.ipynb` (preserved for reference)  
**Refactored Code Structure**: Organized into modular components following ML best practices
