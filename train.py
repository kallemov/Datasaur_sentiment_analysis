import time
import numpy as np
import random
import torch
from utils.train_opt import TrainOptions
from models import create_model
from tqdm import tqdm
from data import  create_training_data
from utils import metrics 
from torch.utils.tensorboard import SummaryWriter

def seed_random(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

if __name__ == '__main__':
    #this code creates an instance for the training 
    opt = TrainOptions().parse()   # get training options

    seed_random(opt.seed)

    #load data for training
    train_data, train_labels, val_data, val_labels = create_training_data(opt)
    
    model = create_model(opt)      # create a model given opt.model and other options
    dataloader_train = model.create_dataloader(opt, train_data, train_labels, randomSample=True)  # create a data_loader
    dataloader_val = model.create_dataloader(opt, val_data, val_labels, randomSample=True)    

    print('The number of training samples = %d' % (len(dataloader_train)*opt.num_epochs))

    
    model.setup(opt, len(dataloader_train))               # regular setup: load and print networks; create schedulers
    
    total_iters = 0                # the total number of training iterations
    writer = SummaryWriter()

    moving_training_loss=0
    
    for epoch in tqdm(range(opt.epoch_start, opt.num_epochs + 1)):
    #for epoch in range(opt.epoch_start, opt.num_epochs + 1):

        epoch_start_time = time.time()  # timer for entire epoch
        loss_train_total = 0
        model.set_train()
        
        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        #progress_bar=dataloader_train
        for batch in progress_bar:
            
            total_iters += opt.batch_size

            model.set_input(batch)         # unpack data from dataset and apply preprocessing
            output = model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            loss=output[0]
            loss_train_total += loss.item()
            moving_training_loss += loss.item()

            if total_iters % opt.print_freq == 0:    # print logging information to the disk
                writer.add_scalar('training loss',moving_training_loss/opt.print_freq, total_iters)
                moving_training_loss=0
                
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('\n saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        tqdm.write(f'\nEpoch {epoch}')
            
        loss_train_avg = loss_train_total/len(dataloader_train)            
        val_loss, predictions, true_vals = model.evaluate(dataloader_val)
       
        print(f'Training loss: {loss_train_avg}')
        print(f'Validation loss: {val_loss}')
        print(f'F1 Score (Weighted): {metrics.f1_score_func(predictions, true_vals)}')
        print(f'AUC Scores: {metrics.auc_score_func(predictions, true_vals)}')
        print(metrics.classification_report_func(predictions, true_vals,opt.label_dict.keys()))
        tqdm.write(f'End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.num_epochs, time.time() - epoch_start_time))
        
        writer.add_scalars('Total loss', {'training loss':loss_train_avg,'validation loss':val_loss}, total_iters)


    
