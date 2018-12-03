import pandas as pd
import numpy as np


#DataGenerator
"""
suppose sort by ROE
Feature Set ={'revenue','liability',take over rate'}
"""

Rev = pd.DataFrame()
Liab = pd.DataFrame()
Take = pd.DataFrame()
ROE = pd.DataFrame()
Feature_Set ={'revenue','liability','take over rate','roe'}

#time index & industry code
#Only 1 industry
time_index = ROE.index.values
company_codes = ROE.index.columns
data = []

for time in time_index:
    temp_data = pd.DataFrame(index = company_codes)
    for item,obj in zip(Feature_Set, [Rev,Liab,Take,ROE]):
        temp_data[item] = obj[time][:,np.newaxis]
    temp_data.set_index(['roe'],inplace = True)
    temp_data.sort_index(ascending = False,inplace = True)

    list_bin = np.linspace(0,1,10)
    temp_data['bin'] = pd.cut(temp_data.index.values,list_bin,labels = list(range(10)))
    temp_data.set_index(['bin'],inplace = True)

    categorical_data = ['revenue','liability']
    ratio_data = ['take over rate']

    #calculate each bin
    final_data = pd.DataFrame(index = list(range(10)))
    for bin in list(range(10)):
        final_data[bin] = np.ravel(temp_data[temp_data.index.values == bin])
    #accumulate data of ['revenue','liabilty']
    for i in range(1,len(temp_data[temp_data.index.values == bin])):
        temp_data.loc[:,[i,i+1]] += temp_data.loc[:,[i-2,i-1]]


    #将binsxfeatures格式的data，展开成一维1x(bins*features)的数据
    temp_data = np.ravel(temp_data)

    data.append(temp_data)

#Input data into CNN
data = np.array(data)    #(times, 10 bins*features)

