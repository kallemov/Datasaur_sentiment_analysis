import time
from utils.predict_opt import PredictOptions
from models import create_model
from data import  create_dataset
from utils.metrics import f1_score_func, accuracy_per_class
import torch

if __name__ == '__main__':
    #this code creates an instance for the training 
    opt = PredictOptions().parse()   # get training options

    seed_random(opt.seed)

    #load data for training
    predict_data = create_dataset(opt)
    
    model = create_model(opt)      # create a model given opt.model and other options
    dataloader = model.create_dataloader(opt, predict_data, randomSample=False)  # create a data_loader
    print('The number of prediction = %d' % len(dataloader))

    
    model.setup(opt, len(dataloader_train))
    start_time = time.time()  # timer for entire epoch
    output = model.predict(dataloader)
    print('Time Taken: %d sec' % time.time() - start_time))

    scores=torch.max(torch.nn.Softmax(dim=1)(output[0]),axis=1)
    
    dic={0:'positive',1:'negative',2:'neutral'}
    [(dic[int(scores[1][i])],round(float(x),3), inputs[i][:50]+ ('...' if len(inputs[i])>50 else '')) for i,x in enumerate(scores[0])]

    
