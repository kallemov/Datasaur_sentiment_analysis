## AutoSAL: Automated Sentiment Analysis Labeling

Bak Kallemov, Insight AI Felllowship AISV20B project.

This is a consulting project for Datasaur.ai .

The goal of this project is to leverage the existing labeling software from Datasaur.ai with automatic pre-labeling based on applying of sentiment analysis of fine-tuned BERT and XLNet classification models.

### Project description


## Requisites

can be found in requirements.txt


### Dependencies

- [Streamlit](streamlit.io)

#### Installation
To install the package above, please run:
```shell
pip install -r requirements
```

## Build Environment

to requests all command line parameters for training and evaluation run

$ python train.py --help


# Training

$ python train.py --dataroot <your_training_data_source> 

The dataloader currently accepts csv and json formats of input files for training

There is a default datasets for binary and multi-class (positive/negative/neutral) sentiment analysis training constructed from three publicly available datasets 

# Prediction

$python predict.py --dataroot <your_prediction_data_source> 

## Configs

## Test
- Include instructions for how to run all tests after the software is installed

# Example

```


```
# Example

# Step 1
# Step 2
```


