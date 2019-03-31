# Skin.ly

In order to run the model, obtain username and password for the skin cancer dataset.

Then, insert that information into the load_data_train_model.sh file.

Run the bash script:

```
./train_model/load_data_train_model.sh
```

(it will take some time). The script will run the following python scripts:

- download_preprocess_providers = download the medicare and medicaid provider dataset
- download_dataset = download the skin cancer
- cnn_model = training cnn model 