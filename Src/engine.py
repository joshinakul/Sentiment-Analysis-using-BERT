from tqdm import tqdm
import torch
import torch.nn as nn

def loss_function(outputs,targets):
    return nn.BCEWithLogitsLoss()(outputs,targets.view(-1,1))

def train(dataLoader,model,optimizer,device,scheduler):
    
    model.train()
    
    for bi,d in tqdm(enumerate(dataLoader),total=len(dataLoader)):
        
        ids = d["ids"].to(device, dtype = torch.long)
        mask = d["mask"].to(device, dtype = torch.long)
        token_type_ids = d["token_type_ids"].to(device, dtype = torch.long)
        sentiment = d["sentiment"].to(device, dtype = torch.float)
        
        optimizer.zero_grad()
        outputs = model(
            ids = ids,
            mask = mask,
            token_type_ids = token_type_ids
        )
        
        loss = loss_function(outputs,sentiment)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
def eval(dataLoader,model,device):
    model.eval()
    final_targets = []
    final_outputs = []
    with torch.no_grad():
        for bi,d in tqdm(enumerate(dataLoader),total=len(dataLoader)):
            ids = d["ids"].to(device, dtype = torch.long)
            mask = d["mask"].to(device, dtype = torch.long)
            token_type_ids = d["token_type_ids"].to(device, dtype = torch.long)
            sentiment = d["sentiment"].to(device, dtype = torch.float)
            
            outputs = model(
                ids = ids,
                mask = mask,
                token_type_ids = token_type_ids
            )

            final_targets.extend(sentiment.cpu().detach().numpy().tolist())
            final_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    
    return final_outputs,final_targets
    
        
        
        