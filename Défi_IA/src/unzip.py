# app.py


import zipfile
with zipfile.ZipFile('input.zip', 'r') as zip_ref:
    zip_ref.extractall('../')