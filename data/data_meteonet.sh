#download forecast data
mkdir -p data/data_meteonet/train/X_forecast/2D_arpege_train
mkdir -p data/data_meteonet/train/X_forecast/2D_arome_train
mkdir -p data/data_meteonet/train/X_forecast/3D_arpege_train

#pour 2D_arpege_train

wget https://meteonet.umr-cnrm.fr/dataset/data/defi_ia_challenge/train/X_forecast/2016/2D_arpege_2016.tar.gz 
wget https://meteonet.umr-cnrm.fr/dataset/data/defi_ia_challenge/train/X_forecast/2017/2D_arpege_2017.tar.gz 

tar -xf ./2D_arpege_2016.tar.gz
tar -xf ./2D_arpege_2017.tar.gz

mv ./2D_arpege_2016/* ./data/data_meteonet/train/X_forecast/2D_arpege_train
mv ./2D_arpege_2017/* ./data/data_meteonet/train/X_forecast/2D_arpege_train

rm -r ./2D_arpege_2016
rm -r ./2D_arpege_2017
rm ./2D_arpege_2016.tar.gz
rm ./2D_arpege_2017.tar.gz

# # pour 3D_arpege_train
wget https://meteonet.umr-cnrm.fr/dataset/data/defi_ia_challenge/train/X_forecast/2016/3D_arpege_2016.tar.gz 
wget https://meteonet.umr-cnrm.fr/dataset/data/defi_ia_challenge/train/X_forecast/2017/3D_arpege_2017.tar.gz 

tar -xf ./3D_arpege_2016.tar.gz 
tar -xf ./3D_arpege_2017.tar.gz 

mv ./3D_arpege_2016/* ./data/data_meteonet/train/X_forecast/3D_arpege_train
mv ./3D_arpege_2017/* ./data/data_meteonet/train/X_forecast/3D_arpege_train

rm -r ./3D_arpege_2016
rm -r ./3D_arpege_2017
rm ./3D_arpege_2016.tar.gz
rm ./3D_arpege_2017.tar.gz


# #pour 2D_arome_train
wget https://meteonet.umr-cnrm.fr/dataset/data/defi_ia_challenge/train/X_forecast/2016/2D_arome_2016.tar.gz 
wget https://meteonet.umr-cnrm.fr/dataset/data/defi_ia_challenge/train/X_forecast/2017/2D_arome_2017.tar.gz 

tar -xf ./2D_arome_2016.tar.gz 
tar -xf ./2D_arome_2017.tar.gz 

mv ./2D_arome_2016/* ./data/data_meteonet/train/X_forecast/2D_arome_train
mv ./2D_arome_2017/* ./data/data_meteonet/train/X_forecast/2D_arome_train

rm -r ./2D_arome_2016
rm -r ./2D_arome_2017
rm ./2D_arome_2016.tar.gz
rm ./2D_arome_2017.tar.gz



# # test
mkdir -p data/data_meteonet/test/3D_arpege
mkdir -p data/data_meteonet/test/2D_arpege
mkdir -p data/data_meteonet/test/2D_arome

# pour 2D_arome_test
wget https://meteonet.umr-cnrm.fr/dataset/data/defi_ia_challenge/test/X_forecast/2D_arome_test.tar.gz 

tar -xf ./2D_arome_test.tar.gz

mv ./2D_arome/* ./data/data_meteonet/test/2D_arome

rm -r ./2D_arome
rm ./2D_arome_test.tar.gz

# # pour 2D_arpege_test
wget https://meteonet.umr-cnrm.fr/dataset/data/defi_ia_challenge/test/X_forecast/2D_arpege_test.tar.gz 

tar -xf ./2D_arpege_test.tar.gz

mv ./2D_arpege/* ./data/data_meteonet/test/2D_arpege

rm -r ./2D_arpege
rm ./2D_arpege_test.tar.gz

# # pour 3D_arpege_test
wget https://meteonet.umr-cnrm.fr/dataset/data/defi_ia_challenge/test/X_forecast/3D_arpege_test.tar.gz 

tar -xf ./3D_arpege_test.tar.gz

mv ./3D_arpege/* ./data/data_meteonet/test/3D_arpege

rm -r ./3D_arpege
rm ./3D_arpege_test.tar.gz