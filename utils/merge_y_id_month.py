import os
import warnings

import pandas as pd

warnings.simplefilter(action='ignore')


def create_day_index(row):
    Id = row['Id']
    day = int(Id.split('_')[1])
    return day


def merge_X_train_Y_train(trainpath):
    X = pd.read_csv(os.path.join(
        trainpath, 'X_train.csv'), header=0, sep=',')
    Y = pd.read_csv(os.path.join(trainpath, 'Y_train.csv'), header=0, sep=',')
    X = X.merge(Y[['Id', 'Ground_truth']], on=['Id'])
    X['date'] = pd.to_datetime(X['date'])
    X['month'] = X['date'].dt.month
    X = X.drop(index=X.index[(X.Ground_truth.isna()) | (X.month.isna())])

    # Save data
    X.to_csv(os.path.join(trainpath, 'X_train.csv'), index=False)


def merge_X_test_Id_month(testpath):
    X = pd.read_csv(os.path.join(testpath, 'X_test.csv'), header=0, sep=',')
    Id_month = pd.read_csv(os.path.join(
        testpath, 'Id_month_test.csv'), header=0, sep=',')
    X['day_index'] = X.apply(create_day_index, axis=1)
    X = X.merge(Id_month, on='day_index', how='left')

    # Save data
    X.to_csv(os.path.join(testpath, 'X_test.csv'), index=False)


def main():
    trainpath = os.path.join("Data", "Train", "Train")
    testpath = os.path.join("Data", "Test", "Test")
    merge_X_train_Y_train(trainpath)
    merge_X_test_Id_month(testpath)


if __name__ == "__main__":
    main()
