# AutoSAIL: Automated Sentiment Analysis Interpretable Labeling

Bak Kallemov, Insight AI Felllowship AISV20B project.

This is a consulting project for Datasaur.ai .

The goal of this project is to leverage Datasaur.ai labeling software with automatic pre-labeling based on pretrained fine-tuned BERT and XLNet classification models for the sentiment analysis. The model uses a Integrated Gradient method to obtain a interpretable result for the highlighting a word importance in the sentiment analysis decisions.

## Project description

Streamlit demo: 
To start it use:
```
streamlit asail_streamlit.py -- <options>
```

![](docs/output_cropped.gif)

### Installation
To install the package above, please run:
```shell
pip install -r requirements
```

## Build Environment

to requests all command line parameters for training and evaluation run

```
 python train.py --help
```

## Training

```
python train.py --dataroot <your_training_data_source> 

```

The dataloader currently accepts csv and json formats of input files for training

There is a default datasets for binary and multi-class (positive/negative/neutral) sentiment analysis training constructed from three publicly available datasets 

## Prediction and testing

```
python predict.py [--dataroot <your_prediction_data_source>] 
```

to get evaluation on test dataset

```
python test.py --dataroot <your_test_datasource>
```

## Running RESTful web service using Flask
To start the server use:

```
python api.py [--<model options>] 
```

The API is desined to receive json requests using route '/application/get_prediction'

POST format:

```
{
  "sentences": [
    {
      "id": 54,
      "text": "A masterpiece four years in the making ."
    }
  ]
}
```

the return json format is 

```
{
  "data": [
    {
      "confidence_score": 0.9998,
      "sentence": {
        "id": 54,
        "token_importances": [0.32, -0.28, 0.66, 1.0, -1.0, 0.71, 0.17, -0.6],
        "tokens": ["a", "masterpiece", "four", "years", "in", "the", "making", "."]
      },
      "sentiment": "positive"
    } 
  ]
}
```
## Running Dockerized Streamlit app

Build docker image 
```
docker build -f docker/Dockerfile -t streamlit .
```

Run streamlit in the container using built image using port:8501
```
docker run -p 8501:8501 streamlit
```


