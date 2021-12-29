![header](https://capsule-render.vercel.app/api?type=waving&color=9999FF&height=200&section=header&text=🌦🌡%20Defi-IA%20:%20Team%20%2B1%20for%20the%20win📈🌞&fontSize=40&animation=twinkling&fontColor=FFFFFF&fontAlignY=30)

<center><div style="font-size:30px ; color: #9999FF">Dorian VOYDIE</br>Thomas FRAMERY</br>Yoann Maarek</div></center>

![footer](https://capsule-render.vercel.app/api?type=waving&color=9999FF&height=150&section=footer&fontSize=50)

# Introduction

## Download the dataset :

You can retrieve the dataset from the kaggle competition. Be sure you have access to the competition and have your credentials in a file called "kaggle.json" stored in ~/.kaggle

```Bash
kaggle competitions download -c defi-ia-2022
```

## Install the requirements :

Rou can read the "requirements.txt" file if you want to create an environment and install manually the libraries required. Or you can run the script :

```Bash
bash install_requirements.sh
```

# Train the Neural Network :

Execute this command in a command prompt directly in the main directory

```Bash
python train.py -data defi-ia-2022 -output Results
```
