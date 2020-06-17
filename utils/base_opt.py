import argparse
import os
import torch
import models
import data

class BaseOptions():
     def __init__(self):
          self.initialized = False

     def initialize(self, parser):
          """Define the common options that are used in both training and test."""
          # basic parameters
          parser.add_argument('--dataroot', required=True, help='path to dataset; expects labeled data for training mode')
          parser.add_argument('--data_samples_max_size', type=int, default=20000, help='max number of samples in dataset to sample')
          parser.add_argument('--sentiment_analysis_type', type=str, default='polarity', help='type for the sentiment analysis classification [ polarity | emmotions | violence | stress] default is polarity')
          parser.add_argument('--disable_word_importance', action='store_true', help='disable word importantce output in the model; Use for perfomance.')
          parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
          parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='temporal model parameters are saved here')
          # model parameters
          parser.add_argument('--model', type=str, default='bert', help='chooses which model to use. [bert | xlnet ]')
          parser.add_argument('--number_sentiments', type=int, default=3, help='# of sentiment classes for classification. Default 3 (positive, negative, neutral')
          parser.add_argument('--seed', type=int, default=1, help='seed for random')
          parser.add_argument('--max_sentence_length', type=int, default=300, help='max length of the input sentence. default is 300')
          
          parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
          parser.add_argument('--load_epoch', type=str, default='latest', help='which epoch to load into the model. Default is to use latest cached model')
          parser.add_argument('--show_progress', type=str, default='tqdm', help='show training progress [none | tqdm |')
          parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
          parser.add_argument('--output_attentions', action='store_true', help='output attentions in bert model')
          parser.add_argument('--captum_visualization', action='store_true', help='visualize word importance')
          parser.add_argument('--num_captum_iterations', type=int, default=100, help='number of iteration for the captum model. default is 100')
          
          self.initialized = True
          return parser

     def gather_options(self):
        if not self.initialized:  # check if it has been initialized
          parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
          parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(self,parser)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

     def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.model)
        if not os.path.exists(expr_dir):
             os.makedirs(expr_dir)

        file_name = os.path.join(expr_dir, '{}_opt.txt'.format('train' if self.isTrain else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

     def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        opt.name = opt.model

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
            
        if opt.sentiment_analysis_type=='polarity':
          if opt.number_sentiments==2:
               opt.label_dict = {'positive':0, 'negative':1}
          elif opt.number_sentiments==3:
               opt.label_dict = {'positive':0, 'negative':1, 'neutral':2}
          opt.inv_label_dict={y:x for x,y in opt.label_dict.items()}
        self.opt = opt
        return self.opt
