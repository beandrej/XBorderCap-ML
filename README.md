# XBorderCap‑ML

Predicting **cross‑border transmission capacities** in the European electricity grid with machine‑learning. This repository contains the code used in the semester project *“Predicting Cross‑Border Transmission Capacities in the European Electricity Grid using different Machine Learning Architectures”* (ETH Zürich × BKW).

> **TL;DR**: We compare four models (Base, MLP “Net”, LSTM, and a **Hybrid residual** model that switches between regression and classification per border). The Hybrid model performs best and, when its NTC predictions are fed into BKW’s long‑term price model, the system‑wide **electricity price MAE improves by up to 20.5%** over BKW’s standard profiles—approaching the theoretical limit obtained with perfect capacities.

---

## Key ideas

- **Two capacity regimes**: Flow‑Based Market Coupling (FBMC) vs. Net Transfer Capacity (NTC). FBMC behaves more continuously; many NTC borders are step‑wise/discrete.
- **Two datasets per regime**:
  - **BL (baseline)** — only inputs compatible with price‑model assumptions (no leakage).
  - **FX (feature‑extended)** — adds past‑day and past‑week averages of prices & capacities.
- **Neighbor filtering + PCA**: Keep only features from countries adjacent to the two border countries; reduce dimensionality to 64 while retaining >93% variance.
- **Scaling**: Feature MinMax scaling; **“max‑so‑far” normalization** for targets to handle upward capacity shifts after upgrades.
- **Hybrid model**: Residual blocks; **classification for discrete NTC borders**, regression otherwise.




---

## Repository structure

```
.
├── lit/                    # literature / notes
├── mappings/               # class/ID maps etc.
├── src/
│   ├── data/               # (place prepared tensors / cache here)
│   ├── results/plots/      # figures generated during runs
│   ├── training/           # training & evaluation scripts
│   ├── utils/              # data prep & helpers
│   ├── config.py           # experiment configuration
│   ├── model.py            # model definitions (Base, Net, LSTM, Hybrid)
│   ├── plot.py             # plotting utilities
│   └── run_euler.slurm     # example SLURM job for cluster runs
├── .gitignore
├── environment.yml
├── requirements.txt
└── README.md
```

> Folder names reflect the current repo; adjust paths if your local tree differs.

---

## Installation

Use Conda (recommended):

```bash
conda env create -f environment.yml
conda activate xbordercap-ml   # or the name defined in environment.yml
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Data

All inputs are **publicly available** (ENTSO‑E, NordPool, ELEXON, NASA MERRA‑2, JAO, EEX Transparency, Wikipedia, …). See the paper for exact fields and scrapers. Place raw/processed data where your scripts expect it (default under `src/data/`); the utilities in `src/utils/` handle cleaning, neighbor‑filtering, feature engineering, and PCA.

### Feature map (high level)
- Electricity profiles (demand & generation by tech)
- Weather (temperature, wind speed, rain, irradiance)
- Country/network parameters (area, population, utilization, congestion counters)
- Cross‑border capacities (hourly NTC and FBMC MAXBEX)
- Prices (hourly DA prices per bidding zone)

Two dataset flavors are built for each border (**BL** and **FX**) and for each regime (**NTC / FBMC**).


<img width="600" height="400" alt="Bildschirmfoto 2025-08-22 um 16 12 44" src="https://github.com/user-attachments/assets/280b6f16-f155-42bc-837a-8af866c845fb" />


---

## Training & evaluation

- Configure defaults in `src/config.py` (paths, splits, seeds, PCA dims, scaler settings).
- Models available: `base`, `net`, `lstm`, `hybrid`.
- For many NTC borders, `--model hybrid --task cls` (classification) works best.
- For FBMC borders, prefer regression.
- To run on a cluster: adapt `src/run_euler.slurm` to your scheduler and submit it.

### Repro settings (used in the paper)
- **Single‑target per border** training
- Fixed seeds; 40 epochs; batch size 128 (Hybrid grid‑searched with small variations)
- MinMax feature scaling + PCA(64); “max‑so‑far” target normalization
- Train/test split ≈ 95/5 of full time span; 20% of train used for validation

---

## Results (short version)

- On **FBMC** borders, the **Hybrid** (residual) model achieves the lowest MAE and is the only model with **positive R² on some borders**, outperforming Base/MLP/LSTM.  
- On **NTC** borders, the Hybrid also wins on average thanks to **classification** of discrete capacity steps (though some borders remain easier with regression).
- Feeding **Hybrid‑predicted NTC** values into BKW’s long‑term price model reduces **price MAE by up to 20.5%** vs. BKW’s standard profiles. Using actual historic NTC sets the practical upper bound (≈22.2%).

<img width="630" height="440" alt="Bildschirmfoto 2025-08-22 um 16 18 50" src="https://github.com/user-attachments/assets/640fa02e-b405-4715-bba6-35be526db814" />

See `src/results/plots` for learning curves, per‑border MAE/R², and qualitative border‑level fits.

---

## Why the Hybrid works

Residual connections + per‑border task selection help the model:  
- **Generalize through shifts** (upgrades/policy changes) thanks to residual pathways.  
- **Classify discrete NTC levels** where regression struggles to “snap” to valid steps.  
- **Regress continuous FBMC patterns** that reflect system‑wide optimizations.

---

## Limitations & notes

- The **backtest** only replaces **NTC** (FBMC MAXBEX needs a reliable down‑scaling to actual capacities; placeholder methods would bias results).
- Some borders changed behavior markedly in late 2024; despite “max‑so‑far”, **down‑shifts** remain hard to model without additional context features (e.g., outages, policy).

---

## Roadmap


- Add a solid **MAXBEX→capacity** scaler and include FBMC in backtests
- Evaluate **temporal convolutional** blocks & **pre‑trained residual** forecasters
- Explore **automatic border task assignment** (meta‑classifier vs. manual split)
- Enrich features with **outage/policy** signals and improved network congestion data

---

## Acknowledgements

This project was carried out with and supported by **BKW (Strategic Market Analysis)** and the **IfA Laboratory at ETH Zürich**. 

---

