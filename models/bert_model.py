import torch
from .base_model import BaseModel


class bertmodel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
	
 

    def set_input(self, input):
	pass

    def forward(self):
	pass



