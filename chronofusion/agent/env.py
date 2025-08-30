# Training loop and evaluation logic.
import numpy as np
import pandas as pd

class FXEnv:
    def __init__(self, higher_tf_data, lower_tf_data, initial_balance=10000):
        self.data = lower_tf_data
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.lot_size = 0.1
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.06


        # Position tracking
        self.current_postion = 0.0
        self.entry_price = None
        self.current_step = 0 

            