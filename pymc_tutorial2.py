# import dependencies
from pandas_datareader import data 
import pandas as pandas_datareader


returns = data.get_data_google('SPY', start='2008-5-1', end='2009-12-1')['Close'].pct_change()
len(returns)