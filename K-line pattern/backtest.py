"""
Backtest on K-line patterns
"""
# coding: utf-8

# # Initialization

import talib as tl
import tushare as ts
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import os
codes = pd.read_csv('index/code_50')
example = str(codes['code'].get_values()[5])

def price(FuncName,code,interval = '15',plot=False):
    #print('The example stock is %s and the code is %d with market size as %d' %(code['name'][0],code['code'][0],code['mktcap'][0]) )
    data = ts.get_k_data(code = example, start='2018-05-25', end = '2018-05-29',ktype= interval, autype='qfq',retry_count=5)
    #plt.Line2D(range(len(data['close'])),data['close'])
    #股票采用每日收盘价绘图
    price = data['close']
    
    if plot:
    #绘制股价走势
        plt.figure(dpi=64,figsize=(12,8))
        plt.plot(price,'k-',
        markerfacecolor='blue',markersize=12) 
        plt.show()
    close = data['close']
    high = data['high']
    low = data['low']
    open = data['open']
    test = np.array(getattr(tl,FuncName)(open,high,low,close))
    
    return (close,high,low,open,test,data)


# # Strategy Implementation

# 通过策略，得到－100则认为应该买入，得到100则认为卖出，在值为0的时候，保持持仓量不变


def mock(Funcname,interval,start,end,code,close,high,low,open,test):
    #初始阶段，假定现金为1000人民币，且持仓量为0
    cash = [1000]
    hold_position = [0]
    for i in range(1,len(test)):
        BuyorSell = test[i]
        #如果技术分析显示买入且持仓量为0，则全仓买入
        if BuyorSell == 100 and hold_position[i-1]==0:
            #计算可以买入多少股票
            num_of_shares = int(cash[i-1]/close[i-1])
            hold_position.append(num_of_shares)
            print('%s  Buying %s shares of stock' %(data['date'][i],num_of_shares))
            cash.append(cash[i-1]-num_of_shares*close[i-1])
        #如果卖出且持仓量大于0，则空仓卖出
        elif BuyorSell == -100 and hold_position[i-1]>0:
            print('%s  Selling %s shares of stock' %(data['date'][i],hold_position[i-1]))
            cash_0 = hold_position[i-1]*close[i-1]
            cash.append(cash_0+cash[i-1])
            hold_position.append(0)
        else:
            cash.append(cash[i-1])
            hold_position.append(hold_position[i-1])
    #绘制资产盈亏图，交易策略图，收益率比较图
    stock = np.multiply(hold_position,close)
    asset = cash + stock
    #资产盈亏图
    #plt.figure(dpi=64,figsize=(12,8))
    fig = plt.figure(figsize=(12,9))
    fig.suptitle('Stock Code: %s   Type: %s \nFunction:%s\nPeriod: %s-%s' %(code,interval,Funcname,start,end),fontsize=15,x=0.53,y=0.96)
    gs = gridspec.GridSpec(3, 1, height_ratios=[3,1,3]) 
    #plt.subplot(311)
    
    ax0 = plt.subplot(gs[0])
    ax0.plot(asset,'b-',markerfacecolor='blue',markersize=12)
    #plt.show()
    #交易策略图
    #plt.figure(dpi=64,figsize=(12,2))
    #plt.subplot(312)
    ax1 = plt.subplot(gs[1])
    ax1.plot(test/100,'b-',markerfacecolor='blue',markersize=12)
    #plt.show()
    #收益率比较图()
    ret_0 = tl.ROC(close,timeperiod=5)
    ret   = tl.ROC(asset,timeperiod=5)
    #plt.figure(dpi=64,figsize=(12,6))
    #plt.subplot(313)
    
    ax2 = plt.subplot(gs[2])
    ax2.plot(ret_0,'b--',markerfacecolor='blue',markersize=12,label='market return')
    ax2.plot(ret,'k-',markerfacecolor='blue',markersize=12,label='strategy return')
    ax2.legend(fontsize='large',loc='best')
    #plt.show()
    #分析形态策略盈亏方差
    
    return fig


# # 批量测试

# 随机选取多支小市值股票，对多个形态进行单独分别分析


stocks = np.random.choice(codes['code'],size=10,replace=True)
valid_names = []
intervals = ['5','15','60']
for stock in stocks:
    for interval in intervals:
        print('interval is %s' %interval)
        for FuncName in FuncList:
            try:
                if not os.path.exists('single'):
                    os.mkdir('single')
                (close,high,low,open,test,data) = price(FuncName,stock,interval=interval)
                if np.sum(test==100)>5:
                    start = np.array(data['date'])[0]
                    end = np.array(data['date'])[-1]
                    fig=mock(FuncName,interval,start,end,stock,close,high,low,open,test)
                    fig.savefig('single/%s_%s_%s.jpg' %(FuncName,interval,stock))
                    if FuncName not in valid_names:
                        valid_names.append(FuncName)
            except:
                raise ValueError('ktype is wrong')


# 结合多个形态分析形成交易策略（粗略地进行简单组合），组合方法定为随机选取三个有效形态进行策略叠加，应用少数服从多数原则，如果同时出现卖空，买入和保持情况，定为采用保持策略。

for i in range(15):
    choices = np.random.choice(valid_names,size=3,replace=True)
    if not os.path.exists('compound'):
        os.mkdir('compound')
    for stock in stocks:
        for interval in ['5']:
            try:
                (close,high,low,open,test0,data) = price(choices[0],stock,interval=interval)
                (close,high,low,open,test1,data) = price(choices[1],stock,interval=interval)
                (close,high,low,open,test2,data) = price(choices[2],stock,interval=interval)
                start = np.array(data['date'])[0]
                end = np.array(data['date'])[-1]
                test = test0+test1+test2
                for i in range(len(test)):
                    if test[i] == 300:
                        test[i] = 100
                    elif test[i] == -300:
                        test[i] = -100
                FuncName = '%s+%s+%s' %(choices[0],choices[1],choices[2])
                fig = mock(FuncName,interval,start,end,stock,close,high,low,open,test)
                fig.savefig('compound/%s_%s_%s.jpg' %(FuncName,interval,stock))
            except:
                raise ValueError('something goes wrong')


FuncList =[
        'CDL2CROWS'
        , 'CDL3BLACKCROWS'
        , 'CDL3INSIDE'
        , 'CDL3LINESTRIKE'
        , 'CDL3OUTSIDE'
        , 'CDL3STARSINSOUTH'
        , 'CDL3WHITESOLDIERS'
        , 'CDLABANDONEDBABY'
        , 'CDLADVANCEBLOCK'
        , 'CDLBELTHOLD'
        , 'CDLBREAKAWAY'
        , 'CDLCLOSINGMARUBOZU'
        , 'CDLCONCEALBABYSWALL'
        , 'CDLCOUNTERATTACK'
        , 'CDLDARKCLOUDCOVER'
        , 'CDLDOJI'
        , 'CDLDOJISTAR'
        , 'CDLDRAGONFLYDOJI'
        , 'CDLENGULFING'
        , 'CDLEVENINGDOJISTAR'
        , 'CDLEVENINGSTAR'
        , 'CDLGAPSIDESIDEWHITE'
        , 'CDLGRAVESTONEDOJI'
        , 'CDLHAMMER'
        , 'CDLHANGINGMAN'
        , 'CDLHARAMI'
        , 'CDLHARAMICROSS'
        , 'CDLHIGHWAVE'
        , 'CDLHIKKAKE'
        , 'CDLHIKKAKEMOD'
        , 'CDLHOMINGPIGEON'
        , 'CDLIDENTICAL3CROWS'
        , 'CDLINNECK'
        , 'CDLINVERTEDHAMMER'
        , 'CDLKICKING'
        , 'CDLKICKINGBYLENGTH'
        , 'CDLLADDERBOTTOM'
        , 'CDLLONGLEGGEDDOJI'
        , 'CDLLONGLINE'
        , 'CDLMARUBOZU'
        , 'CDLMATCHINGLOW'
        , 'CDLMATHOLD'
        , 'CDLMORNINGDOJISTAR'
        , 'CDLMORNINGSTAR'
        , 'CDLONNECK'
        , 'CDLPIERCING'
        , 'CDLRICKSHAWMAN'
        , 'CDLRISEFALL3METHODS'
        , 'CDLSEPARATINGLINES'
        , 'CDLSHOOTINGSTAR'
        , 'CDLSHORTLINE'
        , 'CDLSPINNINGTOP'
        , 'CDLSTALLEDPATTERN'
        , 'CDLSTICKSANDWICH'
        , 'CDLTAKURI'
        , 'CDLTASUKIGAP'
        , 'CDLTHRUSTING'
        , 'CDLTRISTAR'
        , 'CDLUNIQUE3RIVER'
        , 'CDLUPSIDEGAP2CROWS'
        , 'CDLXSIDEGAP3METHODS'
]

