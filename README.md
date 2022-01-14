# DEFI_IA
Git repository for the evaluation of the AI frameworks DEFI IA project 2021-2022

## How to submit your work
You must submit your code as a pull request to this repo.

First make a fork of this repo:
![](images/fork.png)
Use git to clone your code in a local repository.
Create a new branch named with your team number (according to this [file](https://docs.google.com/spreadsheets/d/1UHll3nVPrjPy9EfPd-dGmGaxQGMsWdlEL-FCNmTmgn8/edit#gid=907028874))
Add and comit your code to this branch and push the branch to your forked repository.  
Then go to your fork on github, selct your branch and create a pull request.
![](images/pull_request.png).
Add your team members in the description and create your pull request.  
![](images/pull_request2.png)
Send me an email at bertoin@insa-toulouse.fr to warn me about your pull request.
I'll get back to you so you can be sure that everything is ok.

## What should it contain?

Your work must contain:

* the pdf file for your report.
* a file ```requirement.txt``` containing all the required librairies to run your code.  
Be sure that all the needed librairies are present in this file.
Also verify that no unused library is present in this file.

* a python script `train.py` that will train your model and outputs your final trained model as a pickle file and your predictions on the test data in a csv format.
Your script must take as argument:
    *  `--data_path`: the path to a folder containing all the data files.
    Before executing your script, I will create a folder containing all the unziped files obtained when calling the following command: 
    ```console
    kaggle competitions download -c defi-ia-2022
    ```
    *   `--output_folder`: the path to an input folder where to output your model and predictions.
    * a file README.md describing a little your code (for exemple which file/class does the data preprocessing, which one defines your model ...)

I will call your script with the following command:
```console
python train.py --data_path Data --output_folder Results
``` 
Before running this command I will create a virtual env and install the libraries in the `requirements.txt` file.
I expect your code to run without any bug and to produce the desired outputs.  

Please check that it is the case: create a new virtual environment, clone your repo and run the command:

```console
python train.py --data_path PATH_TO_YOUR_DATA_FOLDER --output_folder PATH_TO_OUTPUT_FOLDER
``` 
If the command does not work for me you won't have the points associated to the coding part of the project.

You will be evaluated on the clarity of your code, I do not expect a single file doing all the work!

---

## Setup

In order to execute our code, you must first create a virtual environment using

```console
py -m pip install -r requirements.txt
```

if you're on Windows, or

```console
python3 -m pip install -r requirements.txt
```

if you're on Unix/macOS. Both assume that you have a previous installation of Python 3.

Next, the downloaded data from Kaggle is saved in a folder called 'Data' with the original structure, which is:

```console
+---Data
|   +---Other
|   |   \---Other
|   +---Presentation_slides
|   |   \---Presentation_slides
|   +---Test
|   |   \---Test
|   |       +---Baselines
|   |       \---X_forecast
|   \---Train
|       \---Train
|           +---Baselines
|           \---X_forecast
```

If you want to download all the data, and do all the preprocessing steps, execute the [preprocess](preprocess.py) script by simply running the command:

```console
python preprocessing.py
```

**before** calling the [train.py](train.py) script. But note that this is an __very__ long process as the data are large.

Otherwise, you may call the training script directly, which retrieves the data that has already been preprocessed.

## Our files

### Preprocessing steps

We decided to create several `utils` files, as there are many preprocessing steps. You can follow them inside the [preprocess.py](preprocess.py) script:

1. First, we download all the data, both from Kaggle and from MeteoNet's server.
2. Then, we open all the .nc files and create a data frame.
3. We follow with feature engineering: we create columns with the daily mean and standard deviation of each day for each station.
4. Then, we fill-in all the missing data of X_stations by interpolating the values using the weighted average data of the closest stations.
5. For the 2D and 3D Arpege data, we compute the seasonal mean, i.e., the mean for each season of the year: winter, spring, summer, and fall, to impute the missing values of this data.
6. Before the final step, we merge the X_stations and the 2D and 3D Arpege files into a single data frame.
7. Finally, we merge the X_train and Y_train data frames, and we create the final data to be used for training.

### MLP model

Our neural network, used in the Kaggle competition is defined inside the [train.py](train.py) script. This script also saves the model and the predictions inside the `Results` folder.

## Results

We obtained our results using a CPU version of tensorflow (2.3.0) on Windows 10. The training step took on average 60 seconds.
