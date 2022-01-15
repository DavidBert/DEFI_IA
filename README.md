# MéTéPasNetBaptiste

ALVAREZ Eloïse - BERJON Pierre - DAURAT Jason - PRADEAU Rémi


# Pipeline avec un modèle XGBoost basique

Le pipeline suivant se limite aux données Kaggle et ne contient pas d'optimisation pour le modèle.

## Installation des librairies utiles pour faire fonctionner le pipeline

```bash
pip install -r requirements
```

## Entraînement du modèle et enregistrement d'un fichier de soumissions et du modèle

```bash
python train.py --data_path PATH_TO_FOLDER --output_folder PATH_TO_RESULT_FOLDER
```

Le chemin `--data_path` doit pointer vers le dossier du défi IA, celui contenant les dossiers Forecast, Other, Presentation_slides, Test et Train.

Le chemin `--output_folder` doit pointer vers un dossier préalablement créé pour contenir la soumission et le modèle.

La soumission aura le nom submission.csv et le modèle sera model.pkl.

## Fonctionnement du code

Le dossier Modules est composé de 4 modules qui ont chacun leur rôle:

    - preprocessing.py: permet de supprimer les colonnes inutilisées et d'assigner des valeurs au nan
    - features.py: permet de concaténer train et test, de regrouper le csv par jour et de calculer les features
    - model.py: permet d'initialiser le modèle, de l'entraîner et de faire des prédictions
    - utils.py: contient des fonctions utiles comme calculer la MAPE, créer le csv de soumissions, sauvegarder le modèle...

Le fichier train.py combine ces 4 modules pour obtenir le résultat voulu.