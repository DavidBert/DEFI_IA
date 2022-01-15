import argparse

from functions import dist_feature, Read_Parquet, paiplot, linear_correlation,non_linear_correlation, boxplot, msno_bar, msno_matrix, msno_dendrogram,msno_heatmap
import sweetviz as sv
import pandas as pd 
import os


test_path_input = '../input/output_data/X_test_stations_prepared.gzip'
train_path_input = '../input/output_data/X_train_stations_prepared.gzip'
output_viz = '../input/output_viz/'
y_train_path = '../input/Train/X_stations/Y_train.csv'

if __name__=='__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default=train_path_input, help='Path to imputed_data_train')
    parser.add_argument('--output_folder', type=str, default=output_viz, help='Path to output_folder')

    args = parser.parse_args()

    data_path = args.data_path
    output_folder = args.output_folder

    train = Read_Parquet(train_path_input)
    y_train = pd.read_csv(y_train_path )

    test = Read_Parquet(test_path_input)
    my_report  = sv.analyze([train,'Train'])
    my_report.show_html(os.path.join(output_viz,'First_train_report.html'))
    my_report  = sv.analyze([test,'Test'])
    my_report.show_html(os.path.join(output_viz,'First_test_report.html'))
    numerical_features = ['wind_speed', 'temp', 'dew_point_temp','humidity', 'wind_direction', 'precip','lat', 'lon','height_sta']
    
    print(f"The html reports for train and test data are in the folder {output_viz}")

    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")

    print("Generating pairplot of training data")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")
    #paiplot
    paiplot(train)


    print("Generating the features's distribution")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")
    ## Feature's distributions


    dist_feature(train, "temp")
    dist_feature(train, "dew_point_temp")
    dist_feature(train, "wind_speed")
    dist_feature(train, "wind_direction")
    dist_feature(train, "precip")
    dist_feature(train, "humidity")
    dist_feature(y_train, "Ground_truth")
    ## Correlation 
    linear_correlation(train)
    non_linear_correlation(train)
    ## Boxplot 
    boxplot(train, 'temp')
    boxplot(train, 'dew_point_temp')
    boxplot(train, 'wind_speed')
    boxplot(train, 'wind_direction')
    boxplot(train, 'precip')
    boxplot(train, 'humidity')
    boxplot(y_train, 'Ground_truth')

    ## Missing values viz
    msno_bar(train)


    msno_matrix(train)


    msno_heatmap(train)

    msno_dendrogram(train)

