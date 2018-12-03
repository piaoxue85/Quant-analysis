import tushare as ts
import talib as tl
import tushare as ts
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import os
import datetime
"""
Use empirical distribution of each pattern
to give weight on the confidence of each prediction
"""

if __name__ == '__main__':
    sz_50 = ts.get_sz50s()
    start = '2017-01-01'
    end = '2018-08-27'
    data = get_stock_data(sz_50, start, end)
    last_year_end = start
    last_year_start = datetime.datetime.strptime(start,'%Y-%m-%d')
    last_year_start = last_year_start.replace(last_year_start.year-3).strftime('%Y-%m-%d')

    # Config
    free_ir = 0.035
    rates = get_rates(sz_50,last_year_start,last_year_end)

    portfolio(data,free_ir,rates)


def portfolio(ports,free_ir=0.035,rates=0):
    asset = 0
    ports_name = ports.keys()
    for port in ports_name:
        asset += two_type(ports[port],port,rate=rates[port])
    return asset

def two_type(data, example,free_ir = 0.035,rate = 1):
    close = data['close']
    high = data['high']
    low = data['low']
    open = data['open']
    # 用CDLBELTHO/D进行测试
    t1 = np.array(tl.CDLHIGHWAVE(open, high, low, close))
    t2 = np.array(tl.CDLHANGINGMAN(open, high, low, close))
    t3 = np.array(tl.CDLDRAGONFLYDOJI(open, high, low, close))
    t4 = np.array(tl.CDLHARAMICROSS(open, high, low, close))
    # t4 = np.minimum(t4,0)
    t5 = np.array(tl.CDLDARKCLOUDCOVER(open, high, low, close))

    test = t1 + t2 + t3 + t5 + t4
    rate = get_rate(close,test)

    cash = [500000]
    hold_position = [0]
    days = int(data.count()['amount'])
    # 画出信号发出时间点
    sig_time = {'time': [], 'state': []}
    price = data['close']
    high = np.max(close)

    for item in range(0, days):
        # 没有日内交易
        if item == 0:
            continue
        # print(item,len(hold_position))
        BuyorSell = test[item - 1]
        if BuyorSell == 100 and hold_position[item - 1] == 0:
            num_of_shares = int(cash[item - 1]*rate / open[item])
            hold_position.append(num_of_shares)
            print('%s  Buying %s shares of stock' % (data.index[item], num_of_shares))
            cash.append(cash[item - 1] - num_of_shares * close[item - 1])

            # 添加做多标记
            sig_time['time'].append(pd.DataFrame([high, 0], [price.index[item], price.index[item]]))
            sig_time['state'].append('r')


        elif BuyorSell == -100 and hold_position[item - 1] > 0:
            # 没有做空交易
            print('%s  Selling %s shares of stock' % (data.index[item], hold_position[item - 1]))
            cash_0 = hold_position[item - 1] * open[item]*rate
            cash.append(cash_0 + cash[item - 1])
            hold_position.append(hold_position[item-1]*(1-rate))

            # 添加做空标记
            sig_time['time'].append(pd.DataFrame([high, 0], [price.index[item], price.index[item]]))
            sig_time['state'].append('b--')

        else:
            cash.append(cash[item - 1]+cash[item-1]*free_ir/365)
            hold_position.append(hold_position[item - 1])
    stock = np.multiply(hold_position, close)
    asset = cash + stock
    price = data['close']

    print('Code: ', example, ' is tested.')
    plt.figure(dpi=64, figsize=(30, 20))
    plt.plot(price, 'k-',
             markerfacecolor='blue', markersize=12)
    # plt.plot(test/5)

    # 绘制操作信号时间点
    for i in range(len(sig_time['time'])):
        plt.plot(sig_time['time'][i], sig_time['state'][i])
    plt.show()
    test = np.array(test)
    print(np.count_nonzero(test))
    return asset[-1]


def get_stock_data(codes,start,end,freq = 'D'):
    #codes = pandas.DataFrame(Tushare)
    #start, end = str
    code = codes['code'].get_values()
    api = ts.get_apis()
    data = {}
    for item in code:
        try:
            source = ts.bar(code=str(item),conn = api,start_date=start,end_date=end,freq = freq)
            data[str(item)] = source
        except:
            print('code %s fails.' %str(item))
    return data


def get_rates(codes,data,start,end):
    rate = {}
    code = codes['code'].get_values()
    for i in code:
        close = data[i]['close']
        high = data[i]['high']
        low = data[i]['low']
        open = data[i]['open']
        # 用CDLBELTHO/D进行测试
        t1 = np.array(tl.CDLHIGHWAVE(open, high, low, close))
        t2 = np.array(tl.CDLHANGINGMAN(open, high, low, close))
        t3 = np.array(tl.CDLDRAGONFLYDOJI(open, high, low, close))
        t4 = np.array(tl.CDLHARAMICROSS(open, high, low, close))
        # t4 = np.minimum(t4,0)
        t5 = np.array(tl.CDLDARKCLOUDCOVER(open, high, low, close))

        test = t1 + t2 + t3 + t5 + t4

        rate[i] = get_rate(close,test)
    return rate



def get_rate(price,signal):
    total = np.count_nonzero(signal)
    correct = 0
    for i in range(total):
        if signal[i] == 1 and price[i+1]>price[i-1]:
            correct+=1
        elif signal[i] == -1 and price[i+1]<price[i-1]:
            correct+=1
    rate = correct/total
    return rate



