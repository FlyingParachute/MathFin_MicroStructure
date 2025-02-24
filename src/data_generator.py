import numpy as np
import pandas as pd
from scipy.stats import expon, gamma, powerlaw
import matplotlib.pyplot as plt

np.random.seed(41)

TOTAL_TIME = 32400
NUM_LEVELS = 5
TICK_SIZE = 0.01
INITIAL_MID_PRICE = 170.96
LAMBDA_GLOBAL = 20
LEVEL_WEIGHTS = [0.5, 0.2, 0.15, 0.1, 0.05]
INTRA_DAY_BOOST = {0: 1.0, 10800: 1.0, 21600: 2.0} # 9:00, 12:00, 15:00

EVENT_PROBS = {'L': 0.45, 'C': 0.45, 'M': 0.10}

def initialize_lob():
    bid_queues = gamma.rvs(a=2, scale=10, size=NUM_LEVELS).tolist()
    ask_queues = gamma.rvs(a=2, scale=10, size=NUM_LEVELS).tolist()
    bid_prices = [INITIAL_MID_PRICE - i * TICK_SIZE for i in range(1, NUM_LEVELS + 1)]
    ask_prices = [INITIAL_MID_PRICE + i * TICK_SIZE for i in range(1, NUM_LEVELS + 1)]
    return bid_queues, ask_queues, bid_prices, ask_prices

def simulate_lob_stylized_facts():
    bid_queues, ask_queues, bid_prices, ask_prices = initialize_lob()
    mid_price = INITIAL_MID_PRICE
    events = []
    t = 0
    prev_events = []
    lambda_entries = []

    while t < TOTAL_TIME:
        lambda_scale = 1.0
        for time, boost in INTRA_DAY_BOOST.items():
            if t >= time:
                lambda_scale = boost

        level = np.random.choice(range(NUM_LEVELS), p=LEVEL_WEIGHTS)
        side = np.random.choice(['bid', 'ask'])
        queues = bid_queues if side == 'bid' else ask_queues
        prices = bid_prices if side == 'bid' else ask_prices
        q_before = queues[level]

        lambda_global_adjusted = LAMBDA_GLOBAL * lambda_scale

        excitation = 0
        for prev_t, prev_type in prev_events[-10:]:
            dt = t - prev_t
            if dt > 0:
                excitation += 0.5 * np.exp(-0.05 * dt)
        lambda_global_final = max(0.1, lambda_global_adjusted + excitation)

        dt = expon.rvs(scale=1/lambda_global_final)
        t += dt

        lambda_entries.append({
            'time': t,
            'lambda': lambda_global_final
        })


        if t > TOTAL_TIME:
            break

        event_type = np.random.choice(['L', 'C', 'M'], p=[EVENT_PROBS['L'], EVENT_PROBS['C'], EVENT_PROBS['M']])

        size = powerlaw.rvs(2.5, loc=1, scale=30, size=1)[0]
        size = max(1, min(30, size))

        if event_type == 'L':
            queues[level] += size
        elif event_type == 'C' and q_before > 0:
            cancel_size = min(size, q_before)
            queues[level] -= cancel_size
        elif event_type == 'M':
            opp_queues = ask_queues if side == 'bid' else bid_queues
            if opp_queues[0] > 0:
                rem_size = size
                while rem_size > 0 and opp_queues:
                    available = opp_queues[0]
                    if rem_size < available:
                        opp_queues[0] -= rem_size
                        rem_size = 0
                    else:
                        rem_size -= available
                        opp_queues[0] = 0
                        
                        if np.random.random() < 0.7:
                            if side == 'bid':
                                mid_price = mid_price + 0.5 * TICK_SIZE
                            else:
                                mid_price = mid_price - 0.5 * TICK_SIZE
                            if side == 'bid':
                                ask_prices = [mid_price + i * TICK_SIZE for i in range(1, NUM_LEVELS + 1)]
                            else:
                                bid_prices = [mid_price - i * TICK_SIZE for i in range(1, NUM_LEVELS + 1)]
                        # 在原地补充被耗尽的订单（基于gamma分布）
                        opp_queues[0] = gamma.rvs(a=2, scale=10)

        events.append({
            'time': t,
            'type': event_type,
            'side': side,
            'level': level + 1,
            'size': size,
            'queue_before': q_before,
            'mid_price': mid_price
        })
        prev_events.append((t, event_type))
        if len(prev_events) > 100:
            prev_events.pop(0)

    lambda_entries = pd.DataFrame(lambda_entries)

    return pd.DataFrame(events), bid_queues, ask_queues, bid_prices, ask_prices, lambda_entries

def compute_stats(df):
    stats = {}
    for level in range(1, NUM_LEVELS + 1):
        level_df = df[df['level'] == level]
        stats[level] = {
            '#L': len(level_df[level_df['type'] == 'L']),
            '#C': len(level_df[level_df['type'] == 'C']),
            '#M': len(level_df[level_df['type'] == 'M']),
            'AES': level_df['size'].mean(),
            'AIT': np.diff(level_df['time']).mean() * 1000 if len(level_df) > 1 else np.nan
        }
    
    unique_times = np.unique(df['time'])
    price_changes = np.diff(df['mid_price'].reindex(unique_times, method='ffill'))
    annualized_volatility = price_changes.std() * np.sqrt(252 * 32400)
    
    stats_df = pd.DataFrame(stats).T
    stats_df.loc['Volatility'] = np.nan
    stats_df.at['Volatility', 'Volatility'] = annualized_volatility
    
    return stats_df

stylized_data, stylized_bid_queues, stylized_ask_queues, stylized_bid_prices, stylized_ask_prices,lambda_list = simulate_lob_stylized_facts()
times = stylized_data['time'].unique()
mid_prices = stylized_data['mid_price'].reindex(times, method='ffill')
hours = (times / 3600)

plt.figure(figsize=(9, 3))
plt.plot(hours, mid_prices, label='Simulated Mid-Price', color='blue', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Mid-Price')
plt.title('Mid-Price Dynamics Over Trading Day')
plt.legend()
time_labels = ['9:00', '10:30', '12:00', '13:30', '15:00', '16:30', '18:00']
time_values = [0, 1.5, 3, 4.5, 6, 7.5, 9]
plt.xticks(time_values, time_labels)
plt.yticks([171.0,171.1,171.2])
plt.axhline(y=171.0, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=171.1, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=171.2, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=6, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=9, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
#plt.savefig('results/images/simul_mid_price_dynamics.png')
plt.show()

resampled_times = np.arange(0, TOTAL_TIME + 1, 1)
resampled_prices = stylized_data['mid_price'].reindex(resampled_times, method='ffill')

rolling_volatility = []
window_size = 600
times = resampled_times

for i in range(0, len(times) - window_size + 1): 
    window_times = times[i:i + window_size]
    window_prices = resampled_prices[window_times]
    window_changes = np.diff(window_prices)
    if len(window_changes) > 0:
        vol = window_changes.std() * np.sqrt(252 * 32400)
        rolling_volatility.append((times[i + window_size // 2], vol))
rolling_vol = pd.DataFrame(rolling_volatility, columns=['time', 'volatility'])

stylized_stats = compute_stats(stylized_data)
print("Stylized Facts Simulated Data Stats (Level 1-5):")
print(stylized_stats)
print("\nRolling Volatility Over Time:")
print(rolling_vol)

plt.figure(figsize=(12, 4))
plt.plot(rolling_vol['time'] / 3600, rolling_vol['volatility'], label='Simulated Volatility',color = 'blue')
plt.xlabel('Time')
time_labels = ['9:00', '10:30', '12:00', '13:30', '15:00', '16:30', '18:00']
time_values = [0, 1.5, 3, 4.5, 6, 7.5, 9]
plt.xticks(time_values, time_labels)
plt.ylabel('Annualized Volatility')
plt.title('Rolling 10-Minute Volatility Over Trading Day')
plt.legend()
plt.grid(True)
#plt.savefig('results/images/simul_volatility.png')
plt.show()

#stylized_data.to_csv('data/simul_data.csv', index=False)
#rolling_vol.to_csv('data/simul_volatility.csv', index=False)