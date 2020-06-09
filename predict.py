import time
from utils.predict_opt import PredictOptions
from models import create_model
from data import  create_predict_data
from utils.metrics import f1_score_func, accuracy_per_class
import torch

if __name__ == '__main__':
    #this code creates an instance for the training 
    opt = PredictOptions().parse()   # get training options

    #load data for training
    predict_data = create_predict_data(opt)
    model = create_model(opt)      # create a model given opt.model and other options
    dataloader = model.create_dataloader(opt, predict_data, None, randomSample=False)  # create a data_loader
    print('The number of prediction = %d' % len(predict_data))

    model.setup(opt, len(dataloader))
    start_time = time.time()  # timer for entire epoch
    scores, total_score = model.predict(dataloader)
    print('Time Taken: %d sec' % (time.time() - start_time))

    dic={0:'positive',1:'negative',2:'neutral'}
    scores=[(dic[x[0]],x[1]) for x in scores]
    overall_sentiment = dic[total_score.index(max(total_score))]
    print(scores, "\t overall sentiment is %s"%overall_sentiment)
