import config
import torch


class BERTData:
   
    def __init__(self, feedback, sentiment):
        self.feedback = feedback
        self.sentiment = sentiment
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        
    def __len__(self):
        return len(self.feedback)
    
    def __getitem__(self, item):
        
        feedback = " ".join(str(self.feedback).split())
        inp = self.tokenizer.encode_plus(
            feedback,
            None,
            add_special_tokens=True,
            max_length=self.max_len
        )
       
        ids = inp["input_ids"]
        mask = inp["attention_mask"]
        token_type_ids = inp["token_type_ids"]
        padding = self.max_len - len(ids)
        
        ids = ids + [0]*padding
        mask = mask + [0]*padding
        token_type_ids = token_type_ids + [0]*padding
        
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "sentiment": torch.tensor(self.sentiment[item], dtype=torch.float)
        }
        