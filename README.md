# Stock Trading Using Q-Learning Reinforcement Learning Strategy

This repository implements a stock trading strategy using Q-Learning reinforcement learning, focusing on the NIFTYBEES ETF. The goal is to learn an optimal buy/sell/hold policy that maximizes portfolio returns over time without using deep learning frameworks.

## Dataset
- *Training Dataset*: Historical price data of NIFTYBEES from 2016 to 2020.
- *Testing Dataset*: Price data from 2021 to 2025.
- *Source*: Downloaded using the `yfinance` API from Yahoo Finance.

## Approach
The approach combines classical reinforcement learning with simplified state-action design for real-time trading. The agent learns to take profitable actions through exploration and experience over historical market movements.

### 1. Feature Engineering:
- *Moving Average (MA5)*: 5-day moving average used to define market trend.
- *Trend Signal (`u`)*: Binary signal (1 if price > MA5, else 0) representing trend.
- *State Representation*: Combination of trend signal and position holding status (cash or ETF).

### 2. Q-Learning Framework:
- *States*: 4 states representing all combinations of trend (u) and holding status (t).
- *Actions*: 
  - `0`: Buy
  - `1`: Sell
  - `2`: Hold
- *Reward Function*: Based on profit/loss from each trade or hold decision.
- *Episodes*: 100 per gamma setting.
- *Q-Table*: 4x3 matrix updated using Bellman Equation.

### 3. Training Setup:
- *Epsilon-Greedy Policy*:
  - `epsilon_start`: 1.0 (exploration)
  - `epsilon_decay`: 0.99
  - `epsilon_min`: 0.15
- *Learning Rate (alpha)*: 0.1
- *Discount Factor (gamma)*: Swept from 0.01 to 1.0
- *Portfolio Simulation*: ₹1,000,000 initial capital with full buy/sell actions.

### 4. Evaluation:
- *Testing Phase*: Uses trained Q-table to simulate trades over 5 years of data.
- *Performance Comparison*: Portfolio value is compared across different gamma settings.
- *Visualizations*: Trends, actions, and returns are plotted for analysis.

## Evaluation Metrics:
The performance of the trading agent was evaluated using:
- *Final Portfolio Value* after 5 years of test trading.
- *Total Return* from initial capital.
- *Optimal Gamma*: Gamma value that gives the highest reward.
- *Q-Table Convergence*: Monitored for stability across episodes.

## Results:
The model achieved competitive results with the following:
- *Best Gamma*: Produced the highest portfolio value through stable policy.
- *Q-Table*: Learned consistent policies for all market conditions.
- *Profitable Trading Strategy*: Even with simple trend features and discrete actions.

## Challenges:
- *Limited Feature Set*: Only short-term trend used; does not capture complex market patterns.
- *No Transaction Costs*: Real-world slippage and fees are not included.
- *Static State Space*: Fixed discrete states can be limiting in volatile markets.

## Inference:
The final policy (Q-table) was used to evaluate trading decisions over unseen data from 2021 to 2025. The strategy executed trades consistently, adapting to market trends and maintaining profitability.

## Model Performance Comparison

Below is a comparison of the portfolio value for different gamma values:

| *Gamma* | *Final Portfolio Value (₹)* |
|--------:|-----------------------------:|
| 0.01    | 1,080,000                    |
| 0.50    | 1,135,000                    |
| 0.77    | **1,180,000 (Best)**         |
| 1.00    | 1,120,000                    |

> *Note*: Values are illustrative. Actual output may vary based on randomness and data split.

## Conclusion:
This Q-Learning stock trading strategy illustrates that even with minimal market knowledge and simple features, reinforcement learning can learn profitable trading policies. It serves as a stepping stone toward more advanced RL-based financial strategies.

## Requirements:
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- tqdm
- yfinance

## Usage:
```bash
# Clone the repository
git clone https://github.com/your-repo/qlearning-stock-trading.git

# Install dependencies
pip install -r requirements.txt

# Run training and testing script
python train_and_test.py
