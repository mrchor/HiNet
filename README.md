# Introduction
<u>H</u>ierarchical <u>i</u>nformation extraction <u>Net</u>work (HiNet)

Source code for paper: <font face=Times New Roman>HiNet: A Novel Multi-Scenario & Multi-Task Learning Approach with Hierarchical Information Extraction</font>

Model architecture:

![avatar](./img/model_architecture.png)


## Requirements

Python >= 3.7  
Tensorflow >= 1.15.0  

## Train and Evaluate Model

```
python -u HiNet.py --config_file_path=../config/hinet_sample_schemas.json --task_type=train
```
You can modify the "FLAGS" parameters as needed.

# Acknowledgement
 - The work is supported by [MeiTuan](https://www.meituan.com).