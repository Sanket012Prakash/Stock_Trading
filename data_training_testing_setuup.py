import yfinance as yf
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==== PARAMETERS ====
num_episodes = 100
epsilon_start = 1.0
epsilon_min = 0.15
epsilon_decay = 0.99  # 1% decay per episode
alpha = 0.1
gamma = 0.95
initial_cash = 1_000_000

# ==== DATA PREPARATION ====
def prepare_data(symbol, start, end):
    # Download data and handle multi-index columns
    df = yf.download(symbol, start=start, end=end)
    
    # Flatten multi-index columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    else:
        df.columns = [col.strip() for col in df.columns.values]
    
    # Select and rename close price column
    close_col = [col for col in df.columns if 'Close' in col][0]
    df = df[[close_col]].copy()
    df.rename(columns={close_col: 'Close'}, inplace=True)
    
    # Calculate indicators
    df['MA5'] = df['Close'].rolling(5).mean()
    df = df.dropna()
    df['u'] = (df['Close'] > df['MA5']).astype(int)
    
    return df

# Training and testing datasets
df_train = prepare_data('NIFTYBEES.NS', '2016-01-01', '2020-12-31')
df_test = prepare_data('NIFTYBEES.NS', '2021-01-01', '2025-12-31')

print("Training Data Sample:")
print(df_train.head())

# ==== Q-LEARNING SETUP ====
def get_state(u, t):
    return u * 2 + t  # u: 0/1, t: 0/1

q_table = np.zeros((4, 3))  # 4 states, 3 actions (Buy, Sell, Hold)

# ==== TRAINING LOOP ====
for episode in tqdm(range(num_episodes), desc='Training Episodes'):
    epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))
    current_t = 0  # 0 = all cash, 1 = all stocks
    current_cash = initial_cash
    current_shares = 0

    for i in range(len(df_train) - 1):
        current_u = df_train['u'].iloc[i]
        state = get_state(current_u, current_t)

        # Valid actions
        if current_t == 0:
            valid_actions = [0, 2]  # Buy, Hold
        else:
            valid_actions = [1, 2]  # Sell, Hold

        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.choice(valid_actions)
        else:
            q_values = q_table[state, valid_actions]
            action = valid_actions[np.argmax(q_values)]

        close_price = df_train['Close'].iloc[i]

        # Execute action
        if action == 0:  # Buy
            shares_to_buy = int(current_cash // close_price)
            cost = shares_to_buy * close_price
            leftover_cash = current_cash - cost
            current_shares += shares_to_buy
            current_cash = leftover_cash
            current_t = 1
        elif action == 1:  # Sell
            proceeds = current_shares * close_price
            current_cash += proceeds
            current_shares = 0
            current_t = 0
        # Hold does not change position

        # Calculate reward as total portfolio value (absolute INR)
        portfolio_value = current_cash + current_shares * close_price
        reward = portfolio_value

        # Next state
        next_u = df_train['u'].iloc[i + 1]
        next_state = get_state(next_u, current_t)

        # Q-learning update
        next_max = np.max(q_table[next_state, [0, 1, 2]])
        q_table[state, action] += alpha * (reward + gamma * next_max - q_table[state, action])

trained_q_table = q_table.copy()
print("Training completed.\n")

# ==== TESTING FUNCTION ====
def test_strategy(q_table, df):
    current_t = 0
    current_cash = initial_cash
    current_shares = 0
    portfolio_values = []
    actions_taken = []

    for i in range(len(df)):
        current_u = df['u'].iloc[i]
        state = get_state(current_u, current_t)

        if current_t == 0:
            valid_actions = [0, 2]
        else:
            valid_actions = [1, 2]

        # Greedy action selection (no exploration)
        q_values = q_table[state, valid_actions]
        action = valid_actions[np.argmax(q_values)]
        actions_taken.append(action)

        close_price = df['Close'].iloc[i]

        if action == 0:  # Buy
            shares_to_buy = int(current_cash // close_price)
            cost = shares_to_buy * close_price
            leftover_cash = current_cash - cost
            current_shares += shares_to_buy
            current_cash = leftover_cash
            current_t = 1
        elif action == 1:  # Sell
            proceeds = current_shares * close_price
            current_cash += proceeds
            current_shares = 0
            current_t = 0
        # Hold does not change position

        portfolio_value = current_cash + current_shares * close_price
        portfolio_values.append(portfolio_value)

    return portfolio_values, actions_taken

# ==== RUN TEST ====
test_portfolio_values, actions_taken = test_strategy(trained_q_table, df_test)

# ==== VISUALIZATION ====
plt.figure(figsize=(14, 7))
plt.plot(df_test.index, test_portfolio_values, label='Portfolio Value', color='blue')
plt.title('Portfolio Value Progression During Testing (2021-2025)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (INR)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Visualize actions on price chart
action_map = {0: 'Buy', 1: 'Sell', 2: 'Hold'}
action_colors = {0: 'green', 1: 'red', 2: 'gray'}
action_dates = df_test.index

plt.figure(figsize=(14, 7))
plt.plot(df_test.index, df_test['Close'], label='NIFTYBEES Close', alpha=0.5)
for act in [0, 1]:
    idxs = [i for i, a in enumerate(actions_taken) if a == act]
    plt.scatter(df_test.index[idxs], df_test['Close'].iloc[idxs], 
                label=action_map[act], color=action_colors[act], marker='o', s=40)
plt.title('Actions Taken on NIFTYBEES Price (2021-2025)')
plt.xlabel('Date')
plt.ylabel('Close Price (INR)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==== OUTPUTS ====
print("Final Q-table after training:")
print(trained_q_table)
print(f"\nFinal portfolio value after testing: ₹{test_portfolio_values[-1]:,.2f}")
