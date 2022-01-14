import os
import warnings

import pandas as pd

warnings.simplefilter(action='ignore')


def merge_X_arpege_train(trainpath, arpege_filename):
    X_station = pd.read_csv(os.path.join(
        trainpath, 'X_train.csv'), header=0, sep=',')
    arpege = pd.read_csv(os.path.join(
        trainpath, arpege_filename), header=0, sep=',')
    arpege.drop(columns=['date', 'number_sta'], inplace=True)
    if 'month' in arpege.columns.values and 'month' in X_station.columns.values:
        arpege.drop(columns=['month'], inplace=True)
    X_train = X_station.merge(arpege, on=['Id'], how='left')

    # Save data
    X_train.to_csv(os.path.join(trainpath, 'X_train.csv'), index=False)


def merge_X_arpege_test(testpath, arpege_filename):
    X_station = pd.read_csv(os.path.join(
        testpath, 'X_test.csv'), header=0, sep=',')
    arpege = pd.read_csv(os.path.join(
        testpath, arpege_filename), header=0, sep=',')
    arpege.drop(columns=['number_sta'], inplace=True)
    if 'month' in arpege.columns.values and 'month' in X_station.columns.values:
        arpege.drop(columns=['month'], inplace=True)
    X_test = X_station.merge(arpege, on=['Id'], how='left')

    # Save data
    X_test.to_csv(os.path.join(testpath, 'X_test.csv'), index=False)


def main():
    trainpath = os.path.join("Data", "Train", "Train")
    merge_X_arpege_train(trainpath, 'arpege2D.csv')
    merge_X_arpege_train(trainpath, 'arpege3D.csv')

    testpath = os.path.join("Data", "Test", "Test")
    merge_X_arpege_test(testpath, 'arpege2D.csv')
    merge_X_arpege_test(testpath, 'arpege3D.csv')


if __name__ == "__main__":
    main()
