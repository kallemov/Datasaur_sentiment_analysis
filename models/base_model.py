import os
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

class BaseModel():

    def __init__(self, opt):
        self.net=None
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.number_sentiments=opt.number_sentiments
        
    @staticmethod
    def modify_commandline_options(self,parser):
        return parser

    def create_dataloader(opt):
        pass

    def set_input(self, input):
        pass

    def forward(self):
        pass

    def optimize_parameters(self):
        pass

    def setup(self, opt, dataset_size):

        if self.isTrain:
            self.optimizer = AdamW(self.net.parameters(),
                                   lr=opt.lr, 
                                   eps=opt.eps_adam)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                             num_warmup_steps=opt.num_warmup_steps,
                                                             num_training_steps=dataset_size*opt.num_epochs)
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.load_epoch)

        #no need to create a special call for eval     
        if not self.isTrain:
            self.net.eval()
            
        self.print_networks(opt.verbose)
        
    def set_train(self):
           self.net.train()

    def predict(self,dataloader):
        pass

    def evaluate(self,dataloader_val):
        pass
            
    def save_networks(self, epoch):
        if issubclass(self.__class__, BaseModel):
            save_filename = '%s_net_%s.pth' % (epoch, str(self.__class__))
            save_path = os.path.join(self.save_dir, save_filename)
            
            torch.save(self.net.state_dict(), save_path)

    def load_networks(self, epoch):
        if issubclass(self.__class__, BaseModel):
            load_filename = '%s_net_%s.pth' % (epoch, str(self.__class__))
            load_path = os.path.join(self.save_dir, load_filename)

            self.net.load_state_dict(torch.load(load_path, map_location=self.device))

    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        if issubclass(self.__class__, BaseModel) and verbose:
            num_params = 0
            for param in self.net.parameters():
                num_params += param.numel()
                
            print(self.net)
            print('[Network %s] Total number of parameters : %.3f M' % (str(self.net.__class__), num_params / 1e6))
            print('-----------------------------------------------')
                
