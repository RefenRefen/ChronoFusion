import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# --- Example candle data (replace this with your real dataframe) ---
data = {
    'Date': pd.date_range("2023-01-01", periods=20, freq="1min"),
    'Open':  [100 + i*0.5 for i in range(20)],
    'High':  [100 + i*0.5 + 0.3 for i in range(20)],
    'Low':   [100 + i*0.5 - 0.2 for i in range(20)],
    'Close': [100 + i*0.5 + (0.1 if i % 2 == 0 else -0.1) for i in range(20)],
    'Volume':[1000 + i*10 for i in range(20)]
}
df = pd.DataFrame(data).set_index("Date")

# --- Parameters ---
update_interval = 1  # seconds per candle

# --- Matplotlib setup ---
fig, ax = plt.subplots(figsize=(10,5))
ax.set_title("Live Candle Simulation")

# keep track of how many candles have been shown
candle_count = [0]  

def update(frame):
    ax.clear()
    ax.set_title("Live Candle Simulation")

    # take first N candles
    n = candle_count[0] + 1
    df_partial = df.iloc[:n]

    # plot them
    mpf.plot(df_partial, type='candle', ax=ax, volume=False, style='charles')

    candle_count[0] = n % len(df)  # loop when reaching end

ani = FuncAnimation(fig, update, interval=update_interval*1000)

plt.show()