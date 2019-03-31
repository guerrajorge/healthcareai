#!/usr/bin/env bash

wget https://data.cms.gov/api/views/85jw-maq9/rows.csv?accessType=DOWNLOAD

python_path="...python interpreter" # path to python interpreter

save_dataset_to="" # directory where dataset will be stored

$python_path download_preprocess_providers.py --data_dir="$save_dataset_to"/dataset

$python_path download_dataset.py \
--account_user=... \
--account_password=... \
--out_dir="$save_dataset_to"

save_model_to="" # directory where trained model will be saved
$python_path cnn_model.py \
--data_dir="$save_dataset_to"/dataset \
--out_dir="$save_model_to"
