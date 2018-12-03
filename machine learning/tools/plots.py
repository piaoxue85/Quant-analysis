import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# plot multi graphs, including both price and suspended price in the same graph
def plot_df(subplots=False, df=None, title='data'):
    if isinstance(df, pd.DataFrame):
        col = df.columns
        if subplots:
            n = len(col)
            f, ax = plt.subplots(n, 1, figsize=(15, 10), sharex=True)
            for i in range(n):
                df.plot(y=col[i], ax=ax[i])
            f.savefig('reports/%s.png'%title)
        else:
            df.plot(y=col, title=title)
            plt.savefig('reports/%s.png'%title)
    elif isinstance(df, pd.Series):
        if title is None:
            df.plot(title = df.name)
        else:
            df.plot(title = title)
        plt.savefig('reports/%s.png'%title)
    plt.show()

def feature_importance_plot(importance_sorted, title):
    df = pd.DataFrame(importance_sorted, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    plt.figure()
    df.plot(kind='barh', x='feature', y='fscore',
            legend=False, figsize=(12, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.tight_layout()
    plt.savefig('reports/xgboost/'+title + '.png', dpi=300)
    plt.show()

def plot_train_test_y (subplots, y_true, y_pred_train, y_pred_test, title):
    plt.clf()
    if subplots:
        fig, ax = plt.subplots(2,1)
        ax[0].plot(range(1,1+len(y_true)), y_true, title = 'true')
        ax[1].plot(range(1, len(y_pred_train)+1), y_pred_train, label = 'predicted_train')
        ax[1].plot(range(1 + len(y_pred_train), len(y_true) + 1), y_pred_test, label='predicted_test')
        ax[1].legend()
        ax[1].title('predicted y')
    else:
        plt.figure()
        plt.plot(range(0,len(y_true)), y_true, label='true')
        plt.plot(range(0, len(y_pred_train)), y_pred_train, label = 'predicted_train')
        plt.plot(range(len(y_pred_train), len(y_true)), y_pred_test, label = 'predicted_test' )
        plt.legend()
    plt.savefig('reports/%s.png'%title)
    plt.show()

