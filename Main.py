import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ticker = 'AAPL'
df = yf.download(ticker, '2020-01-01')
df.Close.plot(figsize=(12, 8))
plt.savefig('AAPL_Close.png')

scaler = StandardScaler()
df['Close'] = scaler.fit_transform(df['Close'])
df.Close
