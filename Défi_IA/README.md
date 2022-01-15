
**Please put the input.zip file in src folder and then run the script unzip.py to unzip the data needed for this work** 

**train.py script can be run by simply running python train.py inside the src folder**

**NOTE : BEFORE RUNING THE SCRIPT train.py please run the command conda env update --name name_env --file meteo_env.yml or pip install -r requirement to install the necessary packages for this work**

This Project is about building a forecast model to predict the rainfall accumulation of the next day based on the state of the weather of the cuurent day.

There are mainly two source of data, the X_stations data that gives us metrics about the weather like temperature and humidity, and meteo france forecast models that give us the forecast of some wheather metrics like those of stations but in addition to other metrics like pression and temperature in different levels.


Unfortunetulley we couldn't get the whole data of meteo france forecast models due to their big size.


Hence the model we build was based on only the stations's data where we apply some feature engineering and some strategies to fill the missing values that represent a big percentage in the stations data until 40 % for some features.


In this repository we have folders :

src : this folder contains the scripts to run the code of training and preprocessing the data, and visualizations

notebooks : contains the explorations of data and visualization

meteo_yaml file : For set up of the environement. 



To excute the script data_exploration1 you need to execute first the script of data_preparation.py 

To excute the script of data_exploration2 you need to execute first the script of data_aggregation.py 

The notebooks contains some of the experiments we made like feature selection and feature engineering


The script train.py is independent and take as input the data imputed and do all the preprocessing step + training 



All the visualizations will be generated in the folder input/out_viz 










