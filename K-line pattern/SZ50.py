# coding: utf-8
import tushare as ts
import talib as tl
import tushare as ts
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import os

# # choose SSE 50 component index stocks as portfolio
sz_50 = ts.get_sz50s()
start = '2017-11-01'
end = '2018-11-01'
data = get_stock_data(sz_50,start,end)
df = ts.get_deposit_rate()
df.sort('data',ascending = True).head(10)

#No day trading, No sell mechanism, free interest rate (FR = 0.35%)
def get_stock_data(codes,start,end,freq = '5MIN'):
    #codes = pandas.DataFrame(Tushare)
    #start, end = str
    print()
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

def two_type(data,example):
    close = data['close']
    high = data['high']
    low = data['low']
    open = data['open']
    #用CDLBELTHO/D进行测试
    t1 = np.array(tl.CDLHIGHWAVE(open,high,low,close))
    t2 = np.array(tl.CDLHANGINGMAN(open,high,low,close))
    t3 = np.array(tl.CDLDRAGONFLYDOJI(open,high,low,close))
    t4 = np.array(tl.CDLHARAMICROSS (open,high,low,close))
    #t4 = np.minimum(t4,0)
    t5=  np.array(tl.CDLDARKCLOUDCOVER (open,high,low,close))
    test = t1+t2+t3+t5+t4

    cash = [500000]
    hold_position = [0]
    days = int(data.count()['amount'])
    #画出信号发出时间点
    sig_time = {'time':[],'state':[]}
    price = data['close']
    high = np.max(close)

    for item in range(0,days):
    #没有日内交易
        if item == 0:
            continue
        #print(item,len(hold_position))
        BuyorSell = test[item-1]
        if BuyorSell == 100 and hold_position[item-1] ==0:
            num_of_shares = int(cash[item-1]/open[item])
            hold_position.append(num_of_shares)
            print('%s  Buying %s shares of stock' %(data.index[item],num_of_shares))
            cash.append(cash[item-1]-num_of_shares*close[item-1])

            #添加做多标记
            sig_time['time'].append(pd.DataFrame([high+0.5,0],[price.index[item],price.index[item]]))
            sig_time['state'].append('r')


        elif BuyorSell == -100 and hold_position[item-1] > 0:
            #没有做空交易
            print('%s  Selling %s shares of stock' %(data.index[item],hold_position[item-1]))
            cash_0 = hold_position[item-1]*open[item]
            cash.append(cash_0+cash[item-1])
            hold_position.append(0)

            #添加做空标记
            sig_time['time'].append(pd.DataFrame([price.max()+0.5,0],[price.index[item],price.index[item]]))
            sig_time['state'].append('b--')

        else:
            cash.append(cash[item-1])
            hold_position.append(hold_position[item-1])
    stock = np.multiply(hold_position,close)
    asset = cash+stock
    price = data['close']
    
    print('Code: ', example,' is tested.')
    plt.figure(dpi=64,figsize=(30,20))
    plt.ylim(price.min()-0.5,price.max()+0.5)
    plt.plot(price,'k-',
    markerfacecolor='blue',markersize=12) 
    #plt.plot(test/5)

    #绘制操作信号时间点
    for i in range(len(sig_time['time'])):
        plt.plot(sig_time['time'][i],sig_time['state'][i])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('fig/MIN1.1-6.1/'+str(example)+'.png')
    plt.show()
    test = np.array(test)
    print(np.count_nonzero(test))
    return asset


result = pd.DataFrame()
for code in data.keys():
    try:
        asset = two_type(data[code],code)
        asset = pd.DataFrame(asset.values,columns = [code],index = asset.index.sort_values(True)[0])
        result = pd.concat([result,asset],axis=1)
    except:
        print(code,' fails to get data.')
        result[code] = 0
    

result.fillna(method = 'bfill',axis=0,inplace = True)
result['all'] = result.sum(axis=1)
plt.figure(dpi = 64,figsize = (30,20))
plt.ylim(result['all'].min()-0.5,result['all'].max()+0.5)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('Price(1e8)',fontsize=20)
plt.title('SZ50 portfolio    2018.6.1-2018.11.1',fontsize= 30,loc='center')
plt.plot(result['all'],'k-',markerfacecolor = 'black',markersize = 12)
plt.savefig('fig/portfolio/MIN_1.1-6.1_SZ.png')


result['return'] = ((result['all'] - result['all'].shift(1))/result['all'].shift(1)).fillna(method = 'bfill')*100


plt.figure(dpi = 64,figsize = (30,20))
plt.ylim(result['return'].min()-0.5,result['return'].max()+0.5)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('Return(%)',fontsize=20)
plt.title('SZ50 portfolio return curve   2018.6.1-2018.11.1',fontsize= 30,loc='center')
plt.plot(result['return'],'k-',markerfacecolor = 'black',markersize = 12)
plt.savefig('fig/portfolio/D_1.1-6.1_SZ_return.png')


result['cul_ret'] = (result['all']-result['all'][0])/result['all'][0]*100

plt.figure(dpi = 64,figsize = (30,20))
plt.ylim(result['cul_ret'].min()-0.5,result['cul_ret'].max()+0.5)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('Return(%)',fontsize=20)
plt.title('SZ50 portfolio cumulative return curve 2018.6.1-2018.11.1',fontsize= 30,loc='center')
plt.plot(result['cul_ret'],'k-',markerfacecolor = 'black',markersize = 12)
plt.savefig('fig/portfolio/D_1.1-6.1_SZ_cul_return.png')

