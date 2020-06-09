import torch
import numpy as np
from .base_model import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

class bertmodel(BaseModel):
    def modify_commandline_options(self,parser):
        parser.add_argument('--output_attentions', action='store_true', help='output attentions in bert model')
        parser.add_argument('--output_hidden_states', action='store_false', help='output hidden states in bert model')
        parser.add_argument('--pretrained_model_name', type=str, default="bert-base-uncased", help='bert pretrained model name')

        return parser

    def __init__(self, opt):
        torch.cuda.empty_cache()
        BaseModel.__init__(self, opt)
        self.net = BertForSequenceClassification.from_pretrained(opt.pretrained_model_name,
                                                                 num_labels=opt.number_sentiments,
                                                                 output_attentions=opt.output_attentions,
                                                                 output_hidden_states=opt.output_hidden_states)	
        self.net.to(self.device)


    def create_dataloader(self,opt, data, labels=None, randomSample=True):
        
        tokenizer = BertTokenizer.from_pretrained(opt.pretrained_model_name, do_lower_case=True)
        encoded_data = tokenizer.batch_encode_plus(
            data, 
            add_special_tokens=True, 
            return_attention_mask=True, 
            pad_to_max_length=True, 
            max_length=opt.max_sentence_length, 
            return_tensors='pt'
        )

        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']
        labels = torch.tensor(labels)

        dataset = TensorDataset(input_ids, attention_masks, labels)
        if randomSample:
            samp=RandomSampler(dataset)
        else:
            samp=SequentialSampler(dataset)
            
        dataloader = DataLoader(dataset, 
                                sampler=samp, 
                                batch_size=opt.batch_size)
        return dataloader
        
    def set_input(self, input):

        batch = tuple(b.to(self.device) for b in input)
        
        self.inputs = {'input_ids':      batch[0],
                       'attention_mask': batch[1],
                       'labels':         batch[2],
        }       

        

    def optimize_parameters(self):
        
        self.net.zero_grad()
        outputs = self.net(**self.inputs)
        
        loss = outputs[0]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)

        self.optimizer.step()
        self.scheduler.step()

        return outputs
        
    def evaluate(self,dataloader_val):

        self.net.eval()
    
        loss_val_total = 0
        predictions, true_vals = [], []
    
        for batch in dataloader_val:
        
            batch = tuple(b.to(self.device) for b in batch)
        
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[2],
            }

            with torch.no_grad():        
                outputs = self.net(**inputs)
            
                loss = outputs[0]
                logits = outputs[1]
                loss_val_total += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = inputs['labels'].cpu().numpy()
                predictions.append(logits)
                true_vals.append(label_ids)
    
        loss_val_avg = loss_val_total/len(dataloader_val) 
        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)
            
        return loss_val_avg, predictions, true_vals
