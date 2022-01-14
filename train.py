# Imports 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import argparse
import os, sys
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
sys.path.append("utils/")


# Imports personal libraries
# import pre_traitement 
# import features_engineering
import training 
import prediction
import pre_traitement
import preprocessing
import model

def get_args(): 
	parser = argparse.ArgumentParser(description="Script to launch defi-ia training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--data_path", type = str, help = "path to a folder containing all the data files") 
	parser.add_argument("--output_folder", type = str,  help = "path to an input folder where to output your model and predictions") 
	parser.add_argument("--no-preprocessing", action = "store_false", dest ='preprocessing')
	parser.add_argument("--preprocessing", action = "store_true", dest ='preprocessing')
	return parser.parse_args()

def main(): 
	args = get_args()
	output_folder = args.output_folder
	data_path  = args.data_path

	# Folder where to save trained model + predictions
	if not os.path.exists(output_folder):
 		os.makedirs(output_folder)
	
	# Option 1 : preprocessing
	if args.preprocessing: #option1
		print("Begin of preprocessing...")
		df_train, df_X_test = preprocessing.preprocessing(data_path = data_path) 
		# Option 2 : download the dataset already preprocessed
	else: 
 		print("We use datasets already preprocessed.")
 		df_train = pd.read_csv(f'{data_path}/df_train.csv')
 		df_X_test = pd.read_csv(f'{data_path}/df_X_test.csv')
		
	
	# Split the training dataset
	X_train, X_val, y_train, y_val = train_test_split(df_train.drop(["Ground_truth"], axis = 1), df_train["Ground_truth"], test_size=0.10, shuffle=True)
	
	# Prepare the datasets
	X_train = X_train.drop(["Id", "number_sta", "date"], axis = 1)
	X_val = X_val.drop(["number_sta", "date", "Id"], axis = 1)
	X_test = df_X_test.drop(["number_sta", "date"], axis = 1)
	
	# Clip the outliers for the response variable
	min_treshold, max_treshold = pre_traitement.find_outliers(y_train, 1.96)
	y_train = y_train.clip(upper = max_treshold)
	
	# Normalisation 
	var_to_fit = ['ff', 't', 'td', 'hu', 'dd', 'precip', 'lat', 'lon', 'height_sta',
       'forecast_2D_arome_ws', 'forecast_2D_arome_p3031',
       'forecast_2D_arome_u10', 'forecast_2D_arome_v10',
       'forecast_2D_arome_t2m', 'forecast_2D_arome_d2m', 'forecast_2D_arome_r',
       'forecast_2D_arome_tp', 'forecast_2D_arome_msl',
       'forecast_2D_arpege_ws', 'forecast_2D_arpege_p3031',
       'forecast_2D_arpege_u10', 'forecast_2D_arpege_v10',
       'forecast_2D_arpege_t2m', 'forecast_2D_arpege_d2m',
       'forecast_2D_arpege_r', 'forecast_2D_arpege_tp',
       'forecast_2D_arpege_msl', 'forecast_arpege_3D_height_20',
       'forecast_arpege_3D_height_100', 'forecast_arpege_3D_height_500',
	'forecast_arpege_3D_height_875', 'forecast_arpege_3D_height_1375',
	'forecast_arpege_3D_height_2000', 'forecast_arpege_3D_height_3000']
	
	sts = StandardScaler()
	sts.fit(X_train[var_to_fit])

	X_train[var_to_fit] = sts.transform(X_train[var_to_fit])
	X_val[var_to_fit] = sts.transform(X_val[var_to_fit])
	X_test[var_to_fit] =  sts.transform(X_test[var_to_fit])

 	# Model training
	print("Model training ...")
	model_trained = training.main(X_train, y_train) 

	# Save the model 
	model_trained.save(output_folder + '/model.h5')
	#model_trained = keras.models.load_model(output_folder  + '/model.h5')
	
	# Evaluation of the model on the validation dataset
	print("Evaluation of the model on the validation ...")
	y_pred = model_trained.predict(X_val)
	print("R2 score : ", metrics.r2_score(y_val, y_pred))
	print("MAPE score : ", model.MAPEVal(y_pred, y_val.to_numpy())[0]) 
	
	# Prediction on test + save 
	print("Prediction on the test...")
	filename = output_folder + "/submission.csv"
	submit = prediction.prediction(model_trained, X_test, filename)
	print("save in the file " + filename)
	
 	


if __name__ == "__main__":
    main()
