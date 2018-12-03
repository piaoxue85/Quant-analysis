import numpy as np
import pandas as pd
def DataGenerator(filename = 'Train_Dst_NoAuction_ZScore_CF_1.txt',test = False):
    with open('1.NoAuction_Zscore/NoAuction_Zscore_Training/'+filename) as f:
        test_size = 4000
        if test:
            new = np.array(f.readline().split('  ')[1:],dtype='float64')[:,np.newaxis][:test_size]
        else:
            new = np.array(f.readline().split('  ')[1:],dtype='float64')[:,np.newaxis]
        hi = True
        count = 0
        while hi:
            try:
                if test:
                    b = np.array(f.readline().split('  ')[1:],dtype='float64')[:,np.newaxis][:test_size]
                else:
                    b = np.array(f.readline().split('  ')[1:],dtype='float64')[:,np.newaxis]
                new = np.concatenate((new,b),axis=1)
                count += 1
            except:
                hi = False
        f.close()
    new = pd.DataFrame(new)
    return new


def Preprocess(data=None, k=5):
    """
    Parameter k: k=1,2,3,5,10
    """
    m, n = data.shape
    bid = data.loc[:, 2:39:4]
    ask = data.loc[:, 0:39:4]
    vol_a = data.loc[:, 1:39:4]
    vol_b = data.loc[:, 3:39:4]
    ask.columns = vol_a.columns
    bid.columns = vol_b.columns
    vol_a.sort_values(list(vol_a.index.values), axis=1, ascending=True, inplace=True)
    vol_b.sort_values(list(vol_b.index.values), axis=1, ascending=False, inplace=True)
    ask = ask.loc[:, vol_a.columns]
    bid = bid.loc[:, vol_b.columns]
    mid = (bid.iloc[:, 0] + ask.iloc[:, 0]) / 2
    mid_change = mid / mid.shift(1) - 1

    ask_column = []
    bid_column = []
    ask_vol_column = []
    bid_vol_column = []
    for i in range(10):
        ask_column.append('a_' + str(i))
        bid_column.append('b_' + str(i))
        ask_vol_column.append('va_' + str(i))
        bid_vol_column.append('vb_' + str(i))
    mid_column = ['mid', 'mc']

    result = pd.concat([ask, bid, vol_a, vol_b, mid, mid_change], axis=1)
    result.columns = np.concatenate([ask_column, bid_column, ask_vol_column, bid_vol_column, mid_column])

    result.iloc[:, :20] = result.iloc[:, :20].values / result['mid'].values[:, np.newaxis] - 1
    for i in range(9):
        result.iloc[:, i + 21] += result.iloc[:, i + 20]
        result.iloc[:, i + 31] += result.iloc[:, i + 30]

    result = result.iloc[1:]

    labels = [1, 2, 3, 5, 10]
    label = data.iloc[:, labels.index(k) - 5]

    return result, label


if __name__ =='__main__':
    from data_processor import DataForModel as Data
    data = DataGenerator(test =True)
    data,label = Preprocess(data)
    data = pd.concat([data,label],axis=1)

    print(data.shape)

    data = Data(data,test_ratio= 1)
    print(data.len_train)

    x,y = data.get_test_batch(300,False)
    print(x.shape)



