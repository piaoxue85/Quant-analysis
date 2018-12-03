import pandas as pd
import numpy as np
from data.data_processor import DataForModel
from timeit import default_timer as timer
from sklearn.metrics import explained_variance_score, mean_squared_error
import xgboost as xgb
from sklearn.model_selection import train_test_split
from tools.plots import plot_train_test_y


def main():
    window_size = 10
    test_ratio = 0.3

    normalise_window = False
    normalise_y = True

    data = pd.read_csv('data/nasdaq100_padding.csv')
    data_processing = DataForModel(data, normalise_y, test_ratio)
    start = timer()

    #split data
    X_train, Y_train = data_processing.get_train_batch(window_size, normalise_window)
    # reshape window dimension into features, to fit into xgboost model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2]))

    # test set
    X_test, Y_test = data_processing.get_test_batch(window_size, normalise_window)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

    time_cost = timer() - start
    print(' processing train and test data by batch method took %f s' %time_cost)


    # get predicted target y
    y_pred_train, y_pred_test = xgboost_predict(X_train,Y_train,X_test,Y_test,test_ratio)

    # denormalization to get predicted price
    y_pred_train_denorm = (np.exp(y_pred_train/100)*data_processing.train_mean).astype(np.float64)
    y_pred_test_denorm  = (np.exp(y_pred_test/100)*data_processing.train_mean).astype(np.float64)


def xgboost_predict(X_train,Y_train,X_test,Y_test,val_ratio):
    start = timer()
    xgb_params = {
        'booster': 'gbtree',
        'objective': 'reg:linear',  # regression task
        'subsample': 0.80,  # 80% of data to grow trees and prevent overfitting
        'colsample_bytree': 0.85,  # 85% of features used
        'eta': 0.1,
        'max_depth': 5,
        'lambda': 10,
        'seed': 42,
        'beta': 0.5}
    print(xgb_params)
    boosting_iterations = 100
    x_dev, x_val, y_dev, y_val = train_test_split(X_train, Y_train, test_size=val_ratio, random_state=42)
    dm_dev = xgb.DMatrix(x_dev,y_dev)
    dm_val = xgb.DMatrix(x_val,y_val)
    watchlist = [(dm_dev, 'train'),(dm_val,'validate')]
    xgb_model = xgb.train(xgb_params, dm_dev, boosting_iterations, evals=watchlist, verbose_eval=10)

    y_pred_test = xgb_model.predict(xgb.DMatrix(X_test))
    time_cost = timer()-start

    print('explained_variance_score:', explained_variance_score(Y_test,y_pred_test))
    print('final_mse: ', mean_squared_error(Y_test, y_pred_test))
    print('xgboost training took {time}s'.format(time=time_cost))

    y_pred_train = xgb_model.predict(xgb.DMatrix(X_train))
    y_true = np.concatenate((Y_train,Y_test))
    plot_train_test_y(False,y_true,y_pred_train,y_pred_test,'xgboost/y')
    return model, y_pred_train,y_pred_test

if __name__ == '__main__':
    main()