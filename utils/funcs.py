import os
import tarfile
import zipfile

import gdown
import matplotlib.pyplot as plt
import pandas as pd
import requests


def download_arpege(output_folder):
    """Download the Arpege data from MeteoNet's server.

    Args:
        output_folder (str): Folder to save the data.
    """
    urls = ['https://meteonet.umr-cnrm.fr/dataset/data/defi_ia_challenge/train/X_forecast/2016/2D_arpege_2016.tar.gz',
            'https://meteonet.umr-cnrm.fr/dataset/data/defi_ia_challenge/train/X_forecast/2016/3D_arpege_2016.tar.gz',
            'https://meteonet.umr-cnrm.fr/dataset/data/defi_ia_challenge/train/X_forecast/2017/2D_arpege_2017.tar.gz',
            'https://meteonet.umr-cnrm.fr/dataset/data/defi_ia_challenge/train/X_forecast/2017/3D_arpege_2017.tar.gz',
            'https://meteonet.umr-cnrm.fr/dataset/data/defi_ia_challenge/test/X_forecast/2D_arpege_test.tar.gz',
            'https://meteonet.umr-cnrm.fr/dataset/data/defi_ia_challenge/test/X_forecast/3D_arpege_test.tar.gz']

    output_files = [url.split('/')[-1] for url in urls]

    train_path = os.path.join(output_folder, 'Train', 'Train', 'X_forecast')
    test_path = os.path.join(output_folder, 'Test', 'Test', 'X_forecast')
    data_folders = [
        train_path, train_path, train_path, train_path,
        test_path, test_path
    ]

    for url, output_file, data_folder in zip(urls, output_files, data_folders):
        print(f'Downloading {output_file}')
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        r = requests.get(url, stream=True)
        with open(os.path.join(data_folder, output_file), 'wb') as f:
            f.write(r.content)

        print(f'Extracting data from {output_file}')
        with tarfile.open(os.path.join(data_folder, output_file), 'r:gz') as tar:
            tar.extractall(data_folder)


def download_gdrive(output_folder):
    """Download the preprocessed data from Google Drive.

    Args:
        output_folder (str): Folder to save the data
    """
    urls = ["https://drive.google.com/uc?id=1YT8MByCh0svSvDTbg1k4ECL-WemnduoM",
            "https://drive.google.com/uc?id=1QmLUUfHwKedW7cVFpmUUcjM-BwG5Q2ku"]

    output_files = [os.path.join(output_folder, 'X_test_final.zip'),
                    os.path.join(output_folder, 'X_train_final.zip')]

    output_csv = [file.replace('.zip', '.csv') for file in output_files]

    for url, output_file, csv in zip(urls, output_files, output_csv):
        if not os.path.exists(csv):
            print(f'Downloading {output_file}.')
            gdown.download(url, output_file, quiet=False)

            with zipfile.ZipFile(output_file, 'r') as zip_ref:
                zip_ref.extractall(output_folder)
            os.remove(output_file)


def plot_history(history):
    """Plot the training and validation loss and accuracy.
    In this case, the loss and the accuracy are both the MAE.

    Args:
        history (keras.callbacks.History): Training history
    """
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(16, 9))
    plt.grid(True)
    plt.plot(hist.epoch, hist.loss, label='Loss')
    plt.plot(hist.epoch, hist.val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.show()


def main():
    pass


if __name__ == '__main__':
    output_folder = '../Data'
    download_gdrive(output_folder)
    download_arpege(output_folder)
