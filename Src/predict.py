import config
import torch
from model import BERTModel
import time
from flask import Flask,request,jsonify

app = Flask(__name__)
model = None
device = "cuda"

def predict(sentance):
    
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    feedback = " ".join(str(sentance).split())
    inp = self.tokenizer.encode_plus(
        feedback,
        None,
        add_special_tokens=True,
        max_length=self.max_len,
        pad_to_max_length=True
    )
       
    ids = inp["input_ids"]
    mask = inp["attention_mask"]
    token_type_ids = inp["token_type_ids"]
        
        
    ids =  torch.tensor(ids, dtype=torch.long).unsqueeze(0),
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0),
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0),
    sentiment = torch.tensor(self.sentiment[item], dtype=torch.float).unsqueeze(0)
    
    ids = ids.to(device, dtype=torch.long)
    token_type_ids = token_type_ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)
    
    outputs = model(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]

@app.route("/",methods = ["POST"])
def sentiment():
    
    feedback = request.json["feed"]
    start_time = time.time()
    positive_prediction = predict(feedback)
    negative_prediction = 1 - positive_prediction
    response = {}
    response["response"] = {
        'positive': str(positive_prediction),
        'negative': str(negative_prediction),
        'feedback': str(feedback),
        'time_taken': str(time.time() - start_time)
    }
    return flask.jsonify(response)
    
    
if __name__=="__main__":
    
    model = BERTModel()
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)
    model.eval()
    app.run()
        