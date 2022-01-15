# Défi-IA 2022

Dans ce fichier README.md, le lecteur trouvera toutes les étapes permettant de :

* recréer l'environnement python dans lequel nous avons mené toutes nos expérimentations numériques via le fichier `requirements.txt` ;

* récupérer les datas pré-traitées par rapport à celles issues de Kaggle (le code de ce traitement sera mis à disposition mais ne sera pas exécuté car c'est un processus long : les fichiers résultants seront mis à disposition via le lien de notre drive) à dézipper dans votre dossier dont le chemin sera stocké dans data_path ;

* mettre en forme ces datas pour les tenseurs d'entrée des différents réseaux de neurones à entraîner ;

* entraîner le groupe de réseaux de neurones permettant ensuite faire une prédiction sur le X_test ;

* fournir le fichier `predictions_ENM_Les_Rainettes.csv` que le lecteur pourra déposer sur le submit de Kaggle pour visualiser le score MAPE de cette prédiction.

Après avoir testé plus d'une vingtaine de réseaux différents, nous avons sélectionné les 4 meilleurs d'entre eux.
Nous avons donc créé un groupe de réseaux de neurones dont chacun des membres prémiums va produire une prédiction à partir du fichier X_test et/ou X_forecast_test : nous avons ensuite généré un fichier final `predictions_ENM_Les_Rainettes.csv` issu de la moyenne de ces différentes prédictions.

## Fichiers présents sur le git :
* `requirements.txt` qui contient tous les packages nécessaires au projet.
* `train.py` programme principal qui va entraîner le groupe de réseaux de neurones et fournir le fichier `predictions_ENM_Les_Rainettes.csv`.
* `fonctions.py` qui contient toutes les fonctions qui vont être appelées et utilisées dans `train.py` et `preprocessing.py`.
* `preprocessing.py` à utiliser UNIQUEMENT sur les [data](https://www.kaggle.com/c/defi-ia-2022/data) de Kaggle si impossibilité de télécharger les fichiers pré-traités.

###################################
## Etape 0 : Prérequis, création et activation de l'environnement
* Un interpréteur Python (version >= 3.7)
* Avec la distribution [python.org](https://www.python.org/)
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
* Avec la distribution [Anaconda](https://www.anaconda.com/products/individual)/[Miniconda](https://docs.conda.io/en/latest/miniconda.html)
  ```bash
  conda env create
  conda activate envpython
  ```
* Avec la distribution [python.org](https://www.python.org/)
  ```bash
  pip install -r requirements.txt
  ```
* Avec la distribution [Anaconda](https://www.anaconda.com/products/individual)/[Miniconda](https://docs.conda.io/en/latest/miniconda.html)

  L'installation des dépendances est faite en même temps que la création de l'environnement.

###################################
## Etape 1 : Préparation du dossier qui va accueillir les fichiers pré-traités
1) Récupérer le zip contenant les fichiers déjà pré-traités sur le lien suivant :  

2) Dézipper tous les fichiers dans un dossier dont vous renseignerez le chemin d'accès lors de l'exécution du **train.py**.

3) ATTENTION la procédure qui suit est à lancer UNIQUEMENT si le lien du téléchargement du zip ne fonctionne pas :

    a) télécharger les fichiers d'origine sur Kaggle via les commandes :
    ```console
    pip install kaggle
    ```
    puis

    ```console
    kaggle competitions download -c defi-ia-2022
    ```
    b) Dézipper tous les fichiers dans un dossier dont vous renseignerez le chemin d'accès lors de l'exécution du `preprocessing.py`.
    
    c) Exécuter la commande suivante pour générer dans votre dossier les data pré-traités pour la suite :
    ```console
    python preprocessing.py --data_path PATH_TO_YOUR_DATA_FOLDER --output_folder PATH_TO_YOUR_DATA_FOLDER
    ```
    
###################################
## Etape 2 : Mise au format des data pré-traitées et entraînement des 4 meilleurs réseaux de neurones du groupe

   L'exécution de la commande suivante va mettre au format les datas pré-traitées afin d'avoir les bonnes dimensions des tenseurs d'entrée des différents réseaux de neurones du groupe.<br>
   Puis elle va les entraîner sur les différentes données de Train et fournira une prédiction sur le X_station_test et/ou X_forecast_test pour chaque réseau.<br>
   Cette procédure se terminera par la génération du fichier final `predictions_ENM_Les_Rainettes.csv` moyenne des 4 prédictions précédentes.

   ```console
    python train.py --data_path PATH_TO_YOUR_DATA_FOLDER --output_folder PATH_TO_YOUR_DATA_FOLDER
    ```