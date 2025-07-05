# Q-Learning Based Stock Trading Strategy (with Optuna Hyperparameter Optimization)

This project implements a **Q-Learning-based trading agent** that learns to trade the Indian stock market ETF — **NIFTYBEES** — using minimal features. It uses **tabular reinforcement learning**, and hyperparameters are tuned using **Optuna**, a powerful Bayesian optimization library.

---

## Problem Statement

Can a simple reinforcement learning agent, with discrete state space and minimal market signal (5-day moving average), learn to make profitable trades in the Indian equity market?

---

## Key Highlights

- **Reinforcement Learning** using Q-Learning (discrete agent)
- **Trading Strategy**: Buy, Sell, Hold decisions on NIFTYBEES
- **Feature**: Simple 5-day moving average-based momentum signal
- **Hyperparameter Tuning**: Optuna optimization for `alpha` (learning rate) and `gamma` (discount factor)
- **Backtested** on real stock data (2016–2025 split)

---

## Algorithm Design

### State Space (`4` states total)

Each state is a combination of:

| Feature         | Values | Description                         |
|----------------|--------|-------------------------------------|
| Momentum `u`   | 0 or 1 | 1 if price > MA5                    |
| Position `t`   | 0 or 1 | 1 if holding stock, else 0          |
| Final state ID | 0–3    | Computed as `u * 2 + t`             |

---

### Action Space

| Action | Description          |
|--------|----------------------|
| 0      | Buy (All-in)         |
| 1      | Sell (All-out)       |
| 2      | Hold (Do nothing)    |

---

## Reward Function

The reward at each step is defined as the **total portfolio value (cash + stock)**, encouraging long-term wealth accumulation.

---

## Data Preparation

| Dataset   | Source     | Range            |
|-----------|------------|------------------|
| Train     | NIFTYBEES  | 2016–2020        |
| Test      | NIFTYBEES  | 2021–2025        |

Data is pulled using the `yfinance` API and cleaned to compute:

- **5-day moving average**
- **Binary momentum signal (`u`)**

---

## Hyperparameter Optimization

### Optimized Parameters:

| Parameter | Range        | Description              |
|-----------|--------------|--------------------------|
| `alpha`   | 0.01 – 0.99  | Learning rate            |
| `gamma`   | 0.01 – 0.99  | Discount factor          |

We use `Optuna` to **maximize** final portfolio value over test set.

---

## ⚙️ How to Run

### Requirements

Install dependencies:

```bash
pip install -r requirements.txt
