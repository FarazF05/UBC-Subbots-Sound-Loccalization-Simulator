# Subbots Hydrophone Localization Simulator

This project simulates a pinger in water and the signals received by a ring of hydrophones.  
It then tries to localize the pinger using:

- simulated time-domain signals at each hydrophone,
- cross-correlation to estimate TDOAs (Δt = t0 − ti),
- a Gauss–Newton optimization loop to estimate the pinger’s position,
- simple plots to show the hydrophones, true pinger position, and estimated bearing over iterations.

Everything is written in plain Python with `numpy`, `scipy`, and `matplotlib`.

---

## 1. Setup (using a virtual environment)

All commands below assume you are in the **project root** (where `requirements.txt` and `src/` live).

### 1.1 Create and activate a venv

'''bash
python3 -m venv .venv
source .venv/bin/activate      # On Linux

### 1.2 Install depenecies
pip install --upgrade pip
pip install -r requirements.txt

### 1.3 Running the simulator from the terminal
PYTHONPATH=src python -m subbots_sim.sim.sim_env

#### What this does
-places the hydrophones in the plane,

-picks random true pinger positions on a ring,

-simulates pulsed signals at each hydrophone,

-runs Gauss–Newton localization,

-animates the estimated bearing over iterations.

### 1.4 Running tests
PYTHONPATH=src pytest

### 1.5 deactivate virtual environment
''' bash
deactivate