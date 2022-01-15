### Execution
python train.py --data_path my_data_path --output_folder my_output_folder

### Lors de l'execution le programme enregistrera dans le dossier output:
1) un array numpy.npy contenant les index des stations qu'on utilisé pour l'apprentissage
2) la base de donnée d'entrainement prétraité, en format numpy et separé en train et valid:
X_train.npy, X_val.npy, y_train.npy, y_val.npy
3) Le scaler adjusté lors de l'entrainement pour son utilisation à l'heure des predictions
4) Le model entrainé
5) le model.history contenant l'evolution de la loss_function
6) un fichier log contenant la valeur final de l'apprentissage ainsi comme le resultat MAPE a la validation

On peut activer/desactiver les differents etapes de la chaine en selectionant True/False dans les modules qu'on veut executer

 