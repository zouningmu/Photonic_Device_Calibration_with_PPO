# Photonic Device Calibration with PPO (MATLAB)

This repository contains two minimal MATLAB examples that use **Proximal Policy Optimization (PPO)** to calibrate
integrated photonic devices:

1. **MZM-PPO** – calibrate a 4-channel Mach–Zehnder Modulator (MZM) by directly optimizing drive currents.
2. **MZI-PPO** – calibrate a 4×4 Mach–Zehnder Interferometer (MZI) network via 15 phase-control currents, using
   multiple test input–output pairs.

The implementations are kept as small as possible and are intended as **reusable modules** that can be plugged into
your own optical simulators or hardware interfaces.

---

## Environment

- MATLAB **R2024b**
- Reinforcement Learning Toolbox **24.2**
- Deep Learning Toolbox (for `dlnetwork`, `layerGraph`, etc.)

No extra Python or third-party dependencies are required.

---

## File Structure

```text
.
├── MZM/
│   ├── MyCustomEnvPPO.m          # MZM environment (4-dim action)
│   ├── train_mzm_ppo.m           # MZM PPO training script
│   └── compute_input.m           # User-defined objective (simulation or hardware hook)
│
├── MZI/
│   ├── MyCustomEnvPPO.m          # MZI environment (15-dim phase control)
│   ├── train_mzi_ppo.m           # MZI PPO training script
│   └── comput_output.m           # User-defined chip I/O interface
│
└── README.md
```

> **Note**: `compute_input.m` and `comput_output.m` are intentionally left simple.
> In real experiments you should replace them with your own optical simulation or hardware measurement code.

---

## MZM Calibration (4-Channel)

**Core idea**: PPO directly searches a 4-dimensional current vector that minimizes a scalar objective
(e.g. output intensity error).  

- State: `[current(4); scalar_output]` (5×1)
- Action: `current(4)` in `[0, 1]`
- Reward: `0.01 - output_val` (the smaller the output, the larger the reward)

### Run

In `MZM/`:

```matlab
>> run('train_mzm_ppo.m')
```

- If `trainedPPO_MZM.mat` exists, the script **loads** the saved agent and continues training.
- Otherwise, it creates a new PPO agent.
- The best found current vector is saved as `pi_current.mat` (`piphasecurrent` variable).

You can customize `compute_input.m` to call:
- a numerical model, or  
- an external instrument driver for a real MZM chip.

---

## MZI Calibration (15-Phase Network)

**Core idea**: PPO optimizes a 15-dim phase-current vector so that, under a set of test inputs, the chip output
matches pre-defined target outputs as closely as possible.

- Test inputs: `testInputs` (4×N, default N=10)
- Target outputs: `targetOutputs` (4×N)
- State: `[phase(15); loss]` (16×1)
- Action: `phase(15)` in `[0,1]` (can later be mapped to 0–10 mA etc.)
- Loss: sum of squared errors over all test input–output pairs
- Reward: `0.01 - loss`

### Run

In `MZI/`:

```matlab
>> run('train_mzi_ppo.m')
```

- If `trainedPPO_MZM_MZI.mat` exists, the script loads the saved agent and continues training.
- Otherwise, it creates a new PPO agent.
- The best phase-current vector found during training is saved as `best_phase_current.mat`.

Before real experiments:

- Replace the dummy `testInputs` / `targetOutputs` with your measured or designed test set.
- Implement `comput_output(input_4, phase_15)` to call your:
  - optical simulator, or  
  - photonic chip control + photodetector readout.

---

## Notes

- The PPO actor uses a Gaussian policy with **mean** and **variance** heads implemented via `dlnetwork` +
  `layerGraph`, which matches Reinforcement Learning Toolbox 24.2 behavior.
- Hyperparameters (network width, learning rates, horizon, etc.) are intentionally simple and can be tuned for
  specific chips or tasks.
- These scripts are meant as **starting templates** for photonic calibration; feel free to modify the reward
  function, observation design, and environment wrapper for your own devices.

---

## Contact

For questions or collaboration:

- Author: Jiashu Li
- Email: jiashuli@smail.nju.edu.cn
