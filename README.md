# CoE Project Documentation

## 1. Project Introduction
The CoE project is a project focused on conducting relevant operations and analyses on specific datasets. Its aim is to obtain results for different datasets by running the corresponding code.

## 2. Environment Requirements
The following are the environmental requirements for running this project:
- Python Version: `python=3.10`
- PyTorch Version: `pytorch=2.1.1`
- DGL Version: `dgl=2.4.0`
- scikit-learn Version: `scikit-learn=1.5.2`
- Other Dependencies: `packaging`, `pandas`

Please ensure that your operating environment meets the above requirements to guarantee the normal operation of the project.

## 3. Datasets and Execution Instructions
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


## 4. Notes
1. Before running the code, make sure that you have correctly installed all the dependencies.
2. Ensure that the datasets are properly prepared and that the path configurations are correct (if there are any path-related settings).
3. If an error occurs during the running process, please check whether the environment meets the requirements and whether the commands are entered correctly. If you have further questions, you can contact the project maintainers for solutions. 