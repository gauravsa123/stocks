import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import tqdm

## Classes and Functions

class StockRSI():
    '''Relative strength index'''
    def __init__(self, name, period):
        self.stock = StockInfo(name, period)
        self.hist = self.stock.hist
        self.get_RSI()

    def get_RSI(self, N=14):
        self.rsi_df = pd.DataFrame()
        self.rsi_df['change'] = [0]*len(self.hist['Close'])

        self.rsi_df['change'] = (self.hist['Close'] - self.hist['Close'].shift(1)).values
        self.rsi_df = pd.concat((self.rsi_df[self.rsi_df>0], self.rsi_df[self.rsi_df<0]), axis=1).fillna(0).abs()
        self.rsi_df.columns = ['gain', 'loss']
        self.rsi_df[['avg gain', 'avg loss']] = np.zeros((len(self.rsi_df), 2))

        self.rsi_df['avg gain'][:N] = self.rsi_df['gain'].iloc[:N].mean()
        self.rsi_df['avg loss'][:N] = abs(self.rsi_df['loss'].iloc[:N].mean())

        for i, (idx, row) in enumerate(self.rsi_df.iloc[N:].iterrows(), N):
            self.rsi_df['avg gain'][i] = ((self.rsi_df.iloc[i-1]['avg gain']*(N-1)) + self.rsi_df.iloc[i]['gain'])/N
            self.rsi_df['avg loss'][i] = abs(((self.rsi_df.iloc[i-1]['avg loss']*(N-1)) + self.rsi_df.iloc[i]['loss'])/N)

        self.rsi_df['RS'] = self.rsi_df['avg gain'] / self.rsi_df['avg loss']
        self.rsi_df['RSI'] = 100 - (100 / (1 + self.rsi_df['RS']))

        return self.rsi_df

    def plot_RSI(self):
        plt.plot(self.hist.index, self.rsi_df['RSI'])
        plt.xticks(rotation=45)
        plt.axhline(40, c='k', linestyle='--')
        plt.axhline(70, c='k', linestyle='--')
        plt.show()



class StockInfo():
    ''' Get stock info and EMA values'''
    def __init__(self, name, period='5y'):
        self.name = name
        self.stock = yf.Ticker(name)
        self.hist = self.stock.history(period=period)
    
    def get_ema(self, N, series):
        k = 2/(N+1)
        ema = np.ones(series.shape)*series.mean()
        ema[:N] = series.values[:N].mean()
        for i, _ in enumerate(series.iloc[N:], N):
            ema[i] = (series.iloc[i]*k) + (ema[i-1] * (1-k))
        return ema

    def plot(self, series):
        for N in [8, 50, 200]:
            col_name = 'ema_' + str(N)
            self.hist[col_name] = self.get_ema(N, series)
            self.hist[col_name].plot()
        series.plot()
        plt.legend()
        plt.show()

    def buy_hold_returns(self, invest_val=100000):
        num_stocks = math.floor(invest_val/self.hist['Close'][0])
        buy_hold_ret = (self.hist['Close'][-1] - self.hist['Close'][0])*num_stocks
        percnt_return = (buy_hold_ret/invest_val)*100
        print('\nGains from the Buy-Hold strategy: {}, with %: {} \n'.format(buy_hold_ret, percnt_return))
        return buy_hold_ret, percnt_return

    
class MACD():
    def __init__(self, name, period, fast=12, slow=26, smooth=9):
        self.name = name
        self.fast = fast
        self.slow = slow
        self.smooth = smooth
        self.stock = StockInfo(name, period)
        self.hist = self.stock.hist
        self.stock_rsi = StockRSI(name, period)
        self.get_macd()
        self.implement_macd_strategy()

    def get_macd(self):
        self.hist['ema_fast'] = self.stock.get_ema(self.fast, self.hist['Close'])
        self.hist['ema_slow'] = self.stock.get_ema(self.slow, self.hist['Close'])

        self.hist['macd'] = self.hist['ema_fast'] - self.hist['ema_slow']
        self.hist['signal'] = self.hist['macd'].ewm(span=self.smooth, adjust=False).mean()
        self.hist['histo'] = self.hist['macd'] - self.hist['signal']
        return	

    
    # def plot(self):
    #     # self.get_macd()
    #     self.hist['macd'][-100:].plot()
    #     self.hist['signal'][-100:].plot()
    #     plt.legend()
    #     plt.show()

    def implement_macd_strategy(self):    
        self.buy_price = np.ones(self.hist['Close'].shape) * np.nan
        self.sell_price = np.ones(self.hist['Close'].shape) * np.nan
        self.macd_flag = np.zeros(self.hist['Close'].shape)
        self.position = np.zeros(self.hist['Close'].shape)
        flag = 0

        for i in range(len(self.hist['macd'])):
            if self.hist['macd'][i] > self.hist['signal'][i]:
                if flag != 1:
                    self.buy_price[i] = self.hist['Close'][i]
                    flag = 1
                    self.macd_flag[i] = flag
            elif self.hist['macd'][i] < self.hist['signal'][i]:
                if flag != -1:
                    self.sell_price[i] = self.hist['Close'][i]
                    flag = -1
                    self.macd_flag[i] = flag

        ## Get Positions
        for i in range(self.position.shape[0]):
            if self.macd_flag[i] == 1:
                self.position[i] = 1
            elif self.macd_flag[i] == -1:
                self.position[i] = 0
            else:
                self.position[i] = self.position[i-1]                
        return

    def plot(self, num=100, buy_price=None, sell_price=None, flag=False):
        plt.figure(figsize=(12,8))
        ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 5, colspan = 1)
        ax2 = plt.subplot2grid((10,1), (5,0), rowspan = 3, colspan = 1)

        ax1.plot(self.hist['Close'][-num:])
        if flag == True:
            ax1.plot(self.hist['Close'].index[-num:], buy_price[-num:], marker = '^', color = 'green', 
                     markersize = 10, label = 'BUY SIGNAL', linewidth = 0)
            ax1.plot(self.hist['Close'].index[-num:], sell_price[-num:], marker = 'v', color = 'r', 
                     markersize = 10, label = 'SELL SIGNAL', linewidth = 0)
            ax1.legend()

        ax1.set_title(self.name, fontsize=25.0, color='b')
        ax1.text(0.2, 0.8, 'RSI = {}'.format(int(self.stock_rsi.rsi_df['RSI'].iloc[-1])), 
                 fontsize=20.0, transform=ax1.transAxes)

        ax2.plot(self.hist['macd'][-num:], color = 'grey', linewidth = 1.5, label = 'MACD')
        ax2.plot(self.hist['signal'][-num:], color = 'skyblue', linewidth = 1.5, label = 'SIGNAL')

        start = len(self.hist['Close']) - num
        for i in range(start, len(self.hist['Close'])):
            if str(self.hist['histo'][i])[0] == '-':
                ax2.bar(self.hist['Close'].index[i], self.hist['histo'][i], color = '#ef5350')
            else:
                ax2.bar(self.hist['Close'].index[i], self.hist['histo'][i], color = '#26a69a')

        plt.legend()
        plt.show()

    def get_returns(self, invest_val=100000):
        num_stocks = math.floor(invest_val/self.hist['Close'][0])

        self.hist['daily_return'] = self.hist['Close'].diff().fillna(0)
        self.hist['macd_return'] = self.hist['daily_return'] * self.position

        total_return = np.sum((self.hist['macd_return']*num_stocks).values)
        percnt_return = (total_return/invest_val)*100
        print('\nGains from the MACD strategy: {}, with %: {} \n'.format(total_return, percnt_return))

        return total_return, num_stocks


####-------------------

def color_func(val):
    color = 'green' if val == 'buy' else 'red' if val == 'sell' else 'yellow'
    return 'background-color: {}'.format(color)	

###============    

# period = '2y'

# # stock_rsi = StockRSI('AFFLE.NS', period)
# # stock_rsi.get_RSI(stock_rsi.hist['Close'])
# # stock_rsi.plot_RSI()

# stock = StockInfo('AFFLE.NS', period)
# stock.buy_hold_returns()
# # stock.plot(stock.hist['Close'])

# macd = MACD('AFFLE.NS', period, 12, 26, 9)
# macd.plot()
# macd.get_returns()