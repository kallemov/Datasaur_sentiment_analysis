import os
import torch
import numpy as np
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from captum.attr import IntegratedGradients
from captum.attr import visualization

class BertModelWrapper(torch.nn.Module):

    def __init__(self, opt, model):
        super(BertModelWrapper, self).__init__()
        self.opt=opt
        self.model = model
        self.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        
    def _compute_bert_outputs(self,model_bert, embedding_output, attention_mask=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output)

            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            extended_attention_mask = extended_attention_mask.to(dtype=next(model_bert.parameters()).dtype) # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(model_bert.config.num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
                head_mask = head_mask.to(dtype=next(model_bert.parameters()).dtype) # switch to fload if need + fp16 compatibility
            else:
                head_mask = [None] * model_bert.config.num_hidden_layers

            encoder_outputs = model_bert.encoder(embedding_output,
                                                 extended_attention_mask,
                                                 head_mask=head_mask)
            sequence_output = encoder_outputs[0]
            sequence_output.to(self.device)
            pooled_output = model_bert.pooler(sequence_output)
            pooled_output.to(self.device)
            outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
            return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

    def forward(self, embeddings):        
        outputs = self._compute_bert_outputs(self.model.bert, embeddings)
        
        pooled_output = outputs[1]
        #pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)
        return torch.softmax(logits, dim=1)[:, 1].unsqueeze(1)

class captum_bert():

    def __init__(self, opt):
        self.opt=opt
        self.vis_data_records_ig = []
        self.isDefined = False
            
    def show_words_importance(self, vis_data_records_ig):
        visualization.visualize_text(vis_data_records_ig)
        
    def setup(self, model):#setup a captum wrapper model based on bert
        self.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.bert_model_wrapper = BertModelWrapper(self.opt, model)
        self.bert_model_wrapper.to(self.device)
        self.ig = IntegratedGradients(self.bert_model_wrapper)
        self.isDefined=True
    

    def interpret_sentence(self, tokens, input_embedding, label=0):

        if not self.isDefined:
            print('word interpretation model is not setup up!')
            exit(1)
        
        vis_data_records_ig = []
            
        # compute attributions and approximation delta using integrated gradients
        attributions, delta = self.ig.attribute(input_embedding, n_steps=self.opt.num_captum_iterations, return_convergence_delta=True)
        
        pred,pred_ind=0.,0
        
        #tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().numpy().tolist())    
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.detach().cpu().numpy()
        attributions[0]=0
        attributions[-1]=0
        max_a, min_a = max(attributions[1:-1]), min(attributions[1:-1])
        if (max_a - min_a)==0:
            attributions /=max_a
        else:
            attributions /= 0.5*(max_a - min_a)
            attributions -= 1+2.*min_a/(max_a - min_a)
            
        #print(attributions)
        if self.opt.captum_visualization:
            #add_attributions_to_visualizer(attributions_ig, tokens, pred, pred_ind, label, delta, vis_data_records_ig)
            self.add_attributions_to_visualizer(attributions, tokens, 
                                                  pred, pred_ind, label, delta, 
                                                  vis_data_records_ig)
        return attributions, vis_data_records_ig 
            
                
    def add_attributions_to_visualizer(self, attributions, tokens, pred, pred_ind, label, delta, vis_data_records):
        # storing couple samples in an array for visualization purposes
        vis_data_records.append(visualization.VisualizationDataRecord(
            attributions[1:-1],
            pred,
            pred_ind,
            label,
            "label",
            attributions.sum(),       
            tokens[1:-1],
            delta
        ))    

            


