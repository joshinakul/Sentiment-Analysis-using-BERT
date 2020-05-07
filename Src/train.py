import config
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
import dataset
import torch
from torch.utils.data import DataLoader
from model import BERTModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import engine
from sklearn.metrics import accuracy_score

def start():
    
    df = pd.read_csv(config.TRAINING_CSV).fillna("none")
    df["sentiment"] = df["sentiment"].apply(
        lambda x : 1 if x == "positive" else 0
    )
    
    df_train, df_test = tts(
        df,
        test_size = 0.2,
        random_state = 42,
         stratify = df["sentiment"].values
    )
    
    df_train.reset_index(inplace = True, drop = True)
    df_test.reset_index(inplace = True, drop = True)
    
    train_dataset = dataset.BERTData(
        feedback = df_train["review"].values,
        sentiment = df_train["sentiment"].values
    )
    
    train_dataLoader = DataLoader(
        train_dataset,
        batch_size = config.BATCH_SIZE_TRAIN,
        num_workers = 4
    )
    
    test_dataset = dataset.BERTData(
        feedback = df_test["review"].values,
        sentiment = df_test["sentiment"].values
    )
    
    test_dataLoader = DataLoader(
        test_dataset,
        batch_size = config.BATCH_SIZE_VALID,
        num_workers = 4
    )
    
    device = torch.device("cuda")
    model = BERTModel()
    model.to(device)
    
    param_optimizers = list(model.named_parameters())
    no_decay = ["bias","LayerNorm.bias","LayerNorm.weight"]
    optimized_params = [
        {'params':[p for n, p in param_optimizers if not any(nd in n for nd in no_decay)],"weight_decay" : 0.001},
        
        {'params':[p for n, p in param_optimizers if any(nd in n for nd in no_decay)],"weight_decay" : 0.0}   
    ]
    
    train_steps = int(len(df_train)/ config.BATCH_SIZE_TRAIN * config.EPOCHS)
    optimzer = AdamW(
        optimized_params,
        lr = 3e-5
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = train_steps
    )
    
    bestAcc = 0
    for epoch in range(config.EPOCHS):
        
        engine.train(train_dataLoader,model,optimizer,device,scheduler)
        outputs, targets = engine.eval(test_dataLoader,model,device)
        outputs = np.array(outputs) >= 0.5
        accuracy = accuracy_score(targets,outputs)
        print(f"Acc: {accuracy}")
        
        if accuracy > bestAcc:
            torch.save(model.state_dict(),config.MODEL_PATH)
            bestAcc = accuracy

if __name__ == "__main__":
    start()
        