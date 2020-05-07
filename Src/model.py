import transformers
import config
import torch.nn as nn

class BERTModel(nn.Module):
    
    def __init__(self):
        
        super(BERTModel,self).__init__():
            self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
            self.bert_drop = nn.Dropout(0.2)
            self.out = nn.Linear(768,1)
            
    def forward(self,ids,mask,token_type_ids):
        
        seq_output,pooled_output = self.bert(
            ids,
            attention_mask = mask,
            token_type_ids = token_type_ids
        )
        
        drop_out = self.bert_drop(pooled_output)
        final_output = self.out(drop_out)
        return final_output