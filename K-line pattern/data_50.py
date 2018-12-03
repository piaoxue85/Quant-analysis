"""
Get No. of listed companies' market capitalization below 500 million
"""
import tushare as ts
import numpy as np
import os
import pandas as pd
code = pd.read_csv('code_50')
code['code'] = code['code']
code = code['code'].get_values()
amount = len(code)
print('总共有%d家上市公司，市值小于50亿' %amount)
fail = []
mode = False
for i in range(amount):
    if not os.path.exists('data_50/5min/%s'%code[i]):
        try:
            num = str.format('%6d'%code[i]).replace(' ','0')
            data = ts.get_k_data(code=num,start='2016-01-01',end='2018-05-25',ktype='5',autype='qfq',retry_count=5)
            data = data.drop(['code','volume'],axis=1)
            data = data.set_index('date')
            data.to_csv('data_50/5min/%s' %num)
        except:
            mode = True
            fail.append(num)
            print('Failed at '+ num +' ...')
count = 0
while len(fail) != 0 and count <= 5:
    for c in fail:
        try:
            data = ts.get_k_data(code=c,start='2016-01-01',end='2018-05-25',ktype='5',autype='qfq',retry_count=5)
            data = data.drop(['code','volume'],axis=1)
            data = data.set_index('date')
            data.to_csv('data50/5min/%s' %c)
            fail.remove(c)
        except:
            print('Second tray: Failed at '+ c +' ...')
    count+=1

