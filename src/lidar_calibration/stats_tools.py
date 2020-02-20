"""
stats_tools.py
============================
useful stats functions
D.T.Milodowski
"""
import numpy as np
# Calculate correction factor due to bias in mean fitted in log-transformed
# parameter space (Baskerville 1972)
def calculate_baskervilleCF(log_y,log_yhat):
    MSE = np.mean((log_y-log_yhat)**2)
    CF = np.exp(MSE/2) # Correction factor due to fitting regression in log-space (Baskerville, 1972)
    return CF
