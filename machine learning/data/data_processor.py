import numpy as np

class DataForModel:
    def __init__(self, data, normalize_y = False, test_ratio = 0.1):
        df = data.copy()
        self.data_train, self.data_test = self.train_test_split(df, test_ratio, normalize_y)
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)

    def train_test_split(self, data, test_ratio, normalize_y):
        train_size = int(len(data)*(1-test_ratio))
        if normalize_y:
            self.train_mean = np.mean(data.iloc[:,-1][:train_size]) # for denormalization
            data['new_y'] = (np.log(data.iloc[:,-1]) - np.log(self.train_mean)) * 100.0
            data.drop(data.columns[-2], axis=1, inplace=True)
        train = data.values[:train_size]
        test = data.values[train_size:]
        return train, test

    def get_test_batch(self, seq_len , normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])
        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows #(145, 50, 6)
        x = data_windows[:, :-1]
        y = data_windows[:, -1, [-1]] # last column of the feature as prediction target
        return x,y


    def get_train_batch(self, seq_len , normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    break
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            #print('single batch x before generate: ', np.array(x_batch).shape)
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [-1]]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
