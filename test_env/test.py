# import pandas as pd
# import mplfinance as mpf
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import time

# # --- Example candle data (replace this with your real dataframe) ---
# data = {
#     'Date': pd.date_range("2023-01-01", periods=20, freq="1min"),
#     'Open':  [100 + i*0.5 for i in range(20)],
#     'High':  [100 + i*0.5 + 0.3 for i in range(20)],
#     'Low':   [100 + i*0.5 - 0.2 for i in range(20)],
#     'Close': [100 + i*0.5 + (0.1 if i % 2 == 0 else -0.1) for i in range(20)],
#     'Volume':[1000 + i*10 for i in range(20)]
# }
# df = pd.DataFrame(data).set_index("Date")

# # --- Parameters ---
# update_interval = 1  # seconds per candle

# # --- Matplotlib setup ---
# fig, ax = plt.subplots(figsize=(10,5))
# ax.set_title("Live Candle Simulation")

# # keep track of how many candles have been shown
# candle_count = [0]  

# def update(frame):
#     ax.clear()
#     ax.set_title("Live Candle Simulation")

#     # take first N candles
#     n = candle_count[0] + 1
#     df_partial = df.iloc[:n]

#     # plot them
#     mpf.plot(df_partial, type='candle', ax=ax, volume=False, style='charles')

#     candle_count[0] = n % len(df)  # loop when reaching end

# ani = FuncAnimation(fig, update, interval=update_interval*1000)

# plt.show()



import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import mplfinance as mpf

def plot_agent_actions(df, actions, title="Agent Decisions on Candles"):
    """
    Plot candlestick chart with agent's actions.
    
    df: DataFrame with columns ['time', 'open', 'high', 'low', 'close']
    actions: list/array of same length as df, each in {0,1,2,3}
    """
    # Make sure time is datetime
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # Prepare candlesticks
    mc = mpf.make_marketcolors(up='g', down='r', edge='inherit', wick='inherit', volume='in')
    s  = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', facecolor="black")

    addplots = []

    # Actions
    buy_idx   = df.index[np.where(np.array(actions) == 1)]
    sell_idx  = df.index[np.where(np.array(actions) == 2)]
    close_idx = df.index[np.where(np.array(actions) == 0)]
    hold_idx  = df.index[np.where(np.array(actions) == 3)]

    # Use price series for positioning markers
    price = df['close']

    addplots.append(mpf.make_addplot(price.loc[buy_idx], type='scatter', markersize=100, marker='^', color='lime'))
    addplots.append(mpf.make_addplot(price.loc[sell_idx], type='scatter', markersize=100, marker='v', color='red'))
    addplots.append(mpf.make_addplot(price.loc[close_idx], type='scatter', markersize=80, marker='x', color='yellow'))
    addplots.append(mpf.make_addplot(price.loc[hold_idx], type='scatter', markersize=50, marker='.', color='cyan'))

    mpf.plot(df, type='candle', style=s, addplot=addplots,
             title=title, ylabel="Price", figratio=(16,9), volume=False)

# Example usage
# Suppose df is your test data with OHLC and 'time'
df = pd.DataFrame({
    "time": pd.date_range("2023-01-01", periods=50, freq="H"),
    "open": np.random.rand(50)*100+100,
    "high": np.random.rand(50)*100+150,
    "low": np.random.rand(50)*100+50,
    "close": np.random.rand(50)*100+100,
})

# Fake actions for demo
actions = np.random.choice([0,1,2,3], size=50)

plot_agent_actions(df, actions)
plt.show()
