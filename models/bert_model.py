import torch
import numpy as np
from .base_model import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from .captum_bert import captum_bert

class bertmodel(BaseModel):
    def modify_commandline_options(self,parser):
        parser.add_argument('--output_attentions', action='store_true', help='output attentions in bert model')
        parser.add_argument('--captum_visualization', action='store_true', help='visualize word importance')
        parser.add_argument('--pretrained_model_name', type=str, default="bert-base-uncased", help='bert pretrained model name')
        parser.add_argument('--num_captum_iterations', type=int, default=100, help='number of iteration for the captum model. default is 100')
        return parser

    def __init__(self, opt):
        torch.cuda.empty_cache()
        BaseModel.__init__(self, opt)
        self.net = BertForSequenceClassification.from_pretrained(opt.pretrained_model_name,
                                                                 num_labels=opt.number_sentiments,
                                                                 output_attentions=opt.output_attentions,
                                                                 output_hidden_states=(not opt.disable_word_importance))	
        self.net.to(self.device)

    def setup_interpretation_model(self):
        self.interpretation_model = captum_bert(self.opt)
        self.interpretation_model.setup(self.net)

    def create_dataloader(self,opt, data, labels=None, randomSample=True):
        
        self.tokenizer = BertTokenizer.from_pretrained(opt.pretrained_model_name, do_lower_case=True)
        encoded_data = self.tokenizer.batch_encode_plus(
            data, 
            add_special_tokens=True, 
            return_attention_mask=True, 
            pad_to_max_length=True, 
            max_length=opt.max_sentence_length, 
            return_tensors='pt'
        )

        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']
        if labels is not None:
            labels = torch.tensor(labels)
            dataset = TensorDataset(input_ids, attention_masks, labels)
        else:
            dataset = TensorDataset(input_ids, attention_masks)
            
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


    def predict(self,dataloader):

        self.net.eval()
    
        predictions  =[]
        attributions =[]
        
        total_score=[0]*self.number_sentiments
        for batch in dataloader:
        
            batch = tuple(b.to(self.device) for b in batch)
        
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
            }

            with torch.no_grad():        
                outputs = self.net(**inputs)
                scores=torch.max(torch.nn.Softmax(dim=1)(outputs[0]),axis=1)
                for i,x in enumerate(scores[0]):
                    ind=int(scores[1][i])
                    predictions.append((ind,round(float(x),4)))
                    total_score[ind] +=float(x)
                    if not self.opt.disable_word_importance:
                        input_ids, embeddings = inputs['input_ids'][i], outputs[1][0][i]
                        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                        try:
                            pad_index=tokens.index('[PAD]')
                            tokens=tokens[:pad_index]
                            embeddings=embeddings[:pad_index,:]
                        except:
                            pass
                        #get attributes for the class ind
                        attribution = self.interpretation_model.interpret_sentence(tokens, embeddings.unsqueeze(0),ind)
                        attributions.append((tokens[1:-1],attribution[1:-1]))
        #print(predictions,total_score)
        return predictions, total_score, attributions
