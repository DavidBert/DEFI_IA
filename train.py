import argparse
import pickle
import pandas as pd
from Modules.preprocessing import Preprocess
from Modules.features import Feature
from Modules.model import Model
from Modules.utils import check_path, make_submission_csv, mape

parser = argparse.ArgumentParser()
parser.add_argument('--data_path')
parser.add_argument('--output_folder')

args = parser.parse_args()
input_path = args.data_path
output_path = args.output_folder

input_path = check_path(input_path)
output_path = check_path(output_path)

print('Reading data...')
train = pd.read_csv(input_path + 'Train/Train/X_station_train.csv')
y = pd.read_csv(input_path + 'Train/Train/Y_train.csv')
test = pd.read_csv(input_path + 'Test/Test/X_station_test.csv')

print('Cleaning datasets...')
cleaner = Preprocess()
train['month'] = train['date'].apply(lambda x: int(x.split(' ')[0].split('-')[1]))
train = cleaner.remove_unused_cols(train)
train = cleaner.clean(train)
test = cleaner.remove_unused_cols(test)
test = cleaner.clean(test)

print('Computing features...')
ft = Feature()
df = ft.concat_train_test(train, test)

df = ft.group_by_day(df)

df['rain'] = ft.compute_rain(df)
df['wd'] = ft.compute_wind_direction(df)
df[['autumn', 'spring', 'summer', 'winter']] = ft.compute_seasons(df)

X_train = df[df['train'] == 1].drop('train', axis=1)
X_test = df[df['train'] == 0].drop('train', axis=1)
X_train = X_train.merge(y[['Id', 'Ground_truth']], how='left', on='Id')
X_train = X_train[X_train['Ground_truth'].notna()]

print('Fitting model...')
xgboost = Model()
xgboost.train(X_train.drop(['Ground_truth', 'Id'], axis=1), X_train['Ground_truth'])

fit_predictions = xgboost.predict(X_train.drop(['Ground_truth', 'Id'], axis=1))
fit_score = mape(X_train['Ground_truth'], fit_predictions)

print(f"MAPE obtenu sur le jeu d'entraînement: {fit_score}")

predictions = xgboost.predict(X_test.drop('Id', axis=1))

submission = make_submission_csv(X_test['Id'], predictions)

print('Saving submission and model...')
submission.to_csv(output_path + 'submission.csv', index=False)
pickle.dump(xgboost.model, open(output_path + 'model.pkl', "wb"))

print('Done!')