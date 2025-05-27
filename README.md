# CoE Project Documentation

## 1. Project Introduction
This repository provides the official implementation of the paper **"Cooperation of Experts: Fusing Heterogeneous Information with Large Margin"**, accepted at **ICML 2025**.


## 2. Environment Requirements
The following are the environmental requirements for running this project:
- Python Version: `python=3.10`
- PyTorch Version: `pytorch=2.1.1`
- DGL Version: `dgl=2.4.0`
- scikit-learn Version: `scikit-learn=1.5.2`
- Other Dependencies: `packaging`, `pandas`


## 3. Datasets and Execution Instructions
### 📥 Dataset Download
All datasets used in this project can be downloaded from the following link:
🔗 [Download Dataset](https://drive.google.com/file/d/1uAOjhUjo0beLlzB7HrZDh-3o89X6kYWA/view?usp=drive_link)
1. **Datasets `acm/dblp/yelp/mag/amazon`**:
If you wish to obtain the results for these datasets, please run the following command:
```bash
python main.py -dataset [dataset_name]
```
For example, if you want to run the `acm` dataset, the command to use is:
```bash
python main.py -dataset acm
```

2. **Datasets `esp/flickr/iapr/nus`**:
For these datasets, to obtain the results, please run the following command:
```bash
python multi-view.py -dataset [dataset_name]
```
For example, if you want to run the `esp` dataset, the command to use is:
```bash
python multi-view.py -dataset esp
```
