![header](https://capsule-render.vercel.app/api?type=waving&color=9999FF&height=300&section=header&text=🌦🌡Team%20%2B1%20for%20the%20win📈🌞&fontSize=50&animation=twinkling&fontAlignY=38&desc=by%20Dorian%20VOYDIE,%20Thomas%20FRAMERY,%20Yoann%20MAAREK&descAlignY=51&descAlign=62&fontColor=FFFFFF)

# Introduction

We limited ourselves to the data present on Kaggle to realize this pipeline. You will find the details of the reasoning in the report entitled REPORT.PDF, in particular on the addition of the data present on the MeteoNet site: https://meteonet.umr-cnrm.fr/

# Execute the script

## 1 Download the dataset :

You can retrieve the dataset from the kaggle competition. Be sure you have access to the competition and have your credentials in a file called "kaggle.json" stored in ~/.kaggle

```Bash
kaggle competitions download -c defi-ia-2022
```

## 2 Install the requirements :

Create a new environment :

```conda
conda create --name myenv python
conda activate myenv
pip install -r requirements.txt
```

Install the required libraried using conda or pip

# 3 Train the model :

Execute this command in a command prompt directly in the main directory

```Bash
python train.py --data_path defi-ia-2022 --output_folder Results
```

![footer](https://capsule-render.vercel.app/api?type=waving&color=9999FF&height=150&section=footer&fontSize=50)
