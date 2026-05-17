import datetime as dt
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import get_data, plot_data

sd=dt.datetime(2008, 1, 1)
ed=dt.datetime(2009, 1, 1)# Read in adjusted closing prices for given symbols, date range

syms = ["GOOG", "AAPL", "GLD", "XOM"]
gen_plot = False

def neg_sharpratio_stats(inputs, prices, sv=1000000,rfr=0.0, sf=252.0):
    normed = prices / prices.iloc[0]
    alloced = normed * inputs
    port_val = (alloced * sv).sum(axis=1)
    daily_rets = port_val.pct_change().iloc[1:]
    sharpe = (np.sqrt(sf) * (daily_rets.mean() - rfr)/ daily_rets.std())
    return -sharpe

dates = pd.date_range(sd, ed)
#print(dates[:5])
prices_all = get_data(syms, dates)  # automatically adds SPY
#print(prices_all.head())
prices = prices_all[syms]  # only portfolio symbols
#print(prices.head())
prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
#print(prices_SPY.head())
#allocs = np.asarray(
#        [0.2, 0.2, 0.3, 0.3]
#    )
# )  # add code here to find the allocations
n = len(syms)
x0 = np.array([1 / n] * n)
bounds = tuple((0.0, 1.0) for s in range(n))
constraints = ({'type': 'eq', 'fun': lambda inputs: np.sum(inputs) - 1.0})

neg_sr_min = minimize(
    neg_sharpratio_stats,
    x0,
    args=prices,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'disp': False}
)
allocs = neg_sr_min.x
sv = 1000000
normed = prices / prices.iloc[0]
normed_SPY = prices_SPY / prices_SPY.iloc[0]
print("SPY normed", normed_SPY)
alloced = normed * allocs
pos_val = alloced * sv
port_val = alloced.sum(axis=1)
#print(normed.head())
print("port_val", port_val.head())
daily_rets = port_val.pct_change().iloc[1:]
#print(daily_rets.head())
cr = port_val.iloc[-1] / port_val.iloc[0] - 1
avg_daily_rets = daily_rets.mean()
std_daily_rets = daily_rets.std()
sr = np.sqrt(252) * (avg_daily_rets / std_daily_rets)
print(sr)
#print(cr)
# find the allocations for the optimal portfolio
# note that the values here ARE NOT meant to be correct for a test case
# Get daily portfolio value
#port_val = prices_SPY  # add code here to compute daily portfolio values
# Compare daily portfolio value with SPY using a normalized plot
#if gen_plot:
    # add code to plot here
df_temp = pd.concat([port_val, normed_SPY], keys=["Portfolio", "SPY"], axis=1)
plot_data(df_temp,title="Daily Portfolio Value and SPY")
plt.savefig("./images/figure1.png")
plt.close()


