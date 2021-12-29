![header](https://capsule-render.vercel.app/api?type=waving&color=9999FF&height=300&section=header&text=🌦🌡Team%20%2B1%20for%20the%20win📈🌞&fontSize=50&animation=twinkling&fontAlignY=38&desc=by%20Dorian%20VOYDIE,%20Thomas%20FRAMERY,%20Yoann%20MAAREK&descAlignY=51&descAlign=62&fontColor=FFFFFF)

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

![footer](https://capsule-render.vercel.app/api?type=waving&color=9999FF&height=150&section=footer&fontSize=50)
