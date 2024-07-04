import numpy as np
import pandas as pd
import torch
from typing import List

def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.
    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


class subDataset(torch.utils.data.Dataset):
    def __init__(self, X_input, X_target):
        super(subDataset, self).__init__()
        self.X_input = X_input
        self.X_target = X_target

    def __len__(self):
        return self.X_input.shape[0]

    def __getitem__(self, idx):
        return self.X_input[idx], self.X_target[idx]


class Datagen():
    def __init__(self, data_path: str, target_col: int, independent_col: List, win_size: int, pre_T: int,
                 train_share: float):
        self.data = pd.read_csv(data_path).to_numpy()
        self.train_share = train_share
        self.val_num = train_share + (1 - train_share) / 2
        self.win_size = win_size
        self.pre_T = pre_T
        self.target_col = target_col
        self.indep_col = independent_col

    def normalize(self, X):
        means = np.mean(X, axis=0, dtype=np.float32)
        stds = np.std(X, axis=0, dtype=np.float32)
        X = (X - means) / (stds + (stds == 0) * .001)
        return X.astype(np.float32), means, stds

    def generate_latent(self):
        row_num = self.data.shape[0]
        n_train = int(row_num * self.train_share)
        n_val = int(row_num * self.val_num)
        n_test = row_num
        train, mean, std = self.normalize(self.data[:n_train])
        val = (self.data[n_train:n_val] - mean) / std
        test = (self.data[n_val:n_test] - mean) / std
        dta_target_train = self.data[:n_train, self.target_col]
        squeeze_y = dta_target_train[::2]
        norm_y, mean_y, std_y = self.normalize(squeeze_y)
        dta_target_val = self.data[n_train:n_val, self.target_col]
        dta_target_test = self.data[n_val:n_test, self.target_col]
        n_tr, n_val, n_te = len(train) - self.pre_T, len(val) - self.pre_T, len(test) - self.pre_T

        X_train, X_val, X_test, Y_train, Y_val, Y_test = [], [], [], [], [], []

        for i in range(self.win_size * 2, n_tr, 2):
            tr_x = train[i - self.win_size * 2:i]
            tr_x = tr_x[::2]
            X_train.append(tr_x)
            tr_y = dta_target_train[i - self.win_size * 2:i + self.pre_T]
            tr_y = tr_y[::2]
            Y_train.append(tr_y)

        for k in range(self.win_size * 2, n_val, 2):
            val_x = val[k - self.win_size * 2:k]
            val_x = val_x[::2]
            X_val.append(val_x)
            val_y = dta_target_val[k - self.win_size * 2:k + self.pre_T]
            arb = val_y[self.win_size * 2:]
            val_y = val_y[:self.win_size * 2]
            val_y = val_y[::2]
            Y_val.append(np.hstack([val_y, arb]))

        for j in range(self.win_size * 2, n_te, 2):
            te_x = test[j - self.win_size * 2:j]
            te_x = te_x[::2]
            X_test.append(te_x)
            te_y = dta_target_test[j - self.win_size * 2:j + self.pre_T]
            arb = te_y[self.win_size * 2:]
            te_y = te_y[:self.win_size * 2]
            te_y = te_y[::2]
            Y_test.append(np.hstack([te_y, arb]))

        X_train = np.array(X_train).astype(np.float32)
        Y_train = np.array(Y_train).astype(np.float32)
        X_val = np.array(X_val).astype(np.float32)
        Y_val = np.array(Y_val).astype(np.float32)
        X_test = np.array(X_test).astype(np.float32)
        Y_test = np.array(Y_test).astype(np.float32)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test, mean_y, std_y

    def generate_arbitarystep2(self):
        row_num = self.data.shape[0]
        n_train = int(row_num * self.train_share)
        n_val = int(row_num * self.val_num)
        n_test = row_num
        train, mean, std = self.normalize(self.data[:n_train])
        val = (self.data[n_train:n_val] - mean) / std
        test = (self.data[n_val:n_test] - mean) / std
        dta_target_train = self.data[:n_train, self.target_col]
        squeeze_y = dta_target_train[::2]
        norm_y, mean_y, std_y = self.normalize(squeeze_y)
        dta_target_val = self.data[n_train:n_val, self.target_col]
        dta_target_test = self.data[n_val:n_test, self.target_col]
        n_tr, n_val, n_te = len(train) - self.pre_T, len(val) - self.pre_T, len(test) - self.pre_T

        X_train, X_val, X_test, Y_train, Y_val, Y_test = [], [], [], [], [], []

        for i in range(self.win_size * 2, n_tr, 2):
            tr_x = train[i - self.win_size * 2:i]
            tr_x = tr_x[::2]
            X_train.append(tr_x)
            tr_y = dta_target_train[i - 2:i + self.pre_T]
            tr_y = tr_y[::2]
            Y_train.append(tr_y)

        for k in range(self.win_size * 2, n_val, 2):
            val_x = val[k - self.win_size * 2:k]
            val_x = val_x[::2]
            X_val.append(val_x)
            val_y = dta_target_val[k:k + self.pre_T]
            Y_val.append(val_y)

        for j in range(self.win_size * 2, n_te, 2):
            te_x = test[j - self.win_size * 2:j]
            te_x = te_x[::2]
            X_test.append(te_x)
            te_y = dta_target_test[j:j + self.pre_T]
            Y_test.append(te_y)

        X_train = np.array(X_train).astype(np.float32)
        Y_train = np.array(Y_train).astype(np.float32)
        X_val = np.array(X_val).astype(np.float32)
        Y_val = np.array(Y_val).astype(np.float32)
        X_test = np.array(X_test).astype(np.float32)
        Y_test = np.array(Y_test).astype(np.float32)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test, mean_y, std_y