import time
from utils.predict_opt import PredictOptions
from models import create_model
from data import  create_training_data
from utils import metrics
import torch

if __name__ == '__main__':
    #this code creates an instance for prediction from dataset
    opt = PredictOptions().parse()   # get training options

    #load data for training
    start_time = time.time()
    predict_data, predict_labels = create_training_data(opt)
    print('create data time taken: %d sec' % (time.time() - start_time))
    start_time = time.time()
    model = create_model(opt)      # create a model given opt.model and other options
    print('create model time taken: %d sec' % (time.time() - start_time))

    start_time = time.time()
    dataloader = model.create_dataloader(opt, predict_data, predict_labels, randomSample=True)  # create a data_loader
    print('create dataloader time taken: %d sec' % (time.time() - start_time))
    
    print('The number of prediction = %d' % len(predict_data))

    model.setup(opt)
    start_time = time.time()  # timer for entire epoch

    #predict sentence sentiments, overall sentiment, word_importance, and visualization
    test_loss, predictions, true_vals = model.evaluate(dataloader)
    print('Time Taken: %d sec' % (time.time() - start_time))
    print(f'Test loss: {test_loss}')
    print(f'F1 Score (Weighted): {metrics.f1_score_func(predictions, true_vals)}')
    print(f'AUC Scores : {metrics.auc_score_func(predictions, true_vals)}')
    print(metrics.classification_report_func(predictions, true_vals,opt.label_dict.keys()))
    
    
