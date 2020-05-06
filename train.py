import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
import torch
import torch.nn as nn
import warnings
from transformers import BertTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import f1_score, preprocess
from model import BertBase, BertDataset

# Define constants
MAX_LEN = 512
BATCH_SIZE = 1
EPOCHS = 2
LR = 3e-5
DEVICE = "cuda"

def main():
    # Initialize BertBase tokenizer to give to objects
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Use preprocess in utils to yield train data from json
    data = preprocess('data/train-v1.1.json')
    context_list = data["context_list"]
    context_map = data["context_map"]
    question_list = data["question_list"]
    answer_list = data["answer_list"]
    
    training_set = BertDataset(question_list, context_list, answer_list, context_map, tokenizer, MAX_LEN)
    # data_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    
    model = BertBase('bert-base-uncased', 768, MAX_LEN).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=int(len(training_set.a_list)/BATCH_SIZE * EPOCHS)
    )

    for _ in range(EPOCHS):
        training_loop(training_set, model, optimizer, DEVICE, scheduler)
    
    # Save trained model
    torch.save(model, 'saves/fine_tuned_model_main')
    
    # Use preprocess in utils to yield test data from json
    data = preprocess('data/dev-v1.1.json')
    context_list = data["context_list"]
    context_map = data["context_map"]
    question_list = data["question_list"]
    answer_list = data["answer_list"]
    
    test_set = BertDataset(question_list, context_list, answer_list, context_map, tokenizer, MAX_LEN)
    
    em, f1 = eval_loop(test_set, model, DEVICE)
    print('\n'+'*'*10+'\nExact Score: '+str(em)+'\nF1 Score: '+str(f1)+'*'*10+'\n')


def training_loop(dataset, model, optimizer, device, scheduler=None):
    model.train()
    
    for i, data in enumerate(dataset):
        input_ids = data['input_ids']
        segment_ids = data['segment_ids']
        mask = data['mask']
        start = data['start_targets']
        end = data['end_targets']
        
        # one-hot for targets
        start_target = torch.zeros(1, MAX_LEN)
        end_target = torch.zeros(1, MAX_LEN)
        start_target[0, start] = 1
        end_target[0, end] = 1
        
        # torch to GPU
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        mask = mask.to(device)
        start_target = start_target.to(device)
        end_target = end_target.to(device)
        
        # zero out gradients
        optimizer.zero_grad()
        
        # inference
        start_logit, end_logit = model(input_ids=input_ids, mask=mask, segment_ids=segment_ids)
        
        # calculate loss and back propogate
        loss = nn.BCEWithLogitsLoss()(start_logit, start_target)*0.5 + nn.BCEWithLogitsLoss()(end_logit, end_target)*0.5
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print('step: ' + str(i) + ', loss: ' + str(loss))
        if scheduler is not None:
            scheduler.step()

    print('*'*10+'\nEpoch Complete\n'+'*'*10)


def eval_loop(dataset, model, device):
    model.eval()
    
    exact_scores, f1_scores = [], []
    for i, data in enumerate(dataset):        
        input_ids = data['input_ids']
        segment_ids = data['segment_ids']
        mask = data['mask']
        start_targets = data['start_targets']
        end_targets = data['end_targets']
        
        # torch to GPU
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        mask = mask.to(device)
        
        # inference
        start_logit, end_logit = model(input_ids=input_ids, mask=mask, segment_ids=segment_ids)
        _, start_pred = torch.max(start_logit[0], 0)
        _, end_pred = torch.max(end_logit[0], 0)
        
        # cover if prediction has end token before start token
        if end_pred < start_pred:
            start_pred, end_pred = end_pred, start_pred
 
        # calculate best exact and f1 score for prediction among viable answers
        best_exact, best_f1 = 0., 0.
        for i in range(len(start_targets)):
            answer_span = range(int(start_targets[i]), int(end_targets[i]) + 1)
            pred_span = range(int(start_pred), int(end_pred) + 1)
            if pred_span is answer_span:
                best_exact = 1.
            
            f1 = f1_score(pred_span, answer_span)
            if f1 > best_f1:
                best_f1 = f1
                
        exact_scores.append(best_exact)
        f1_scores.append(best_f1)
    
    final_exact = sum(exact_scores) / len(exact_scores)
    final_f1 = sum(f1_scores) / len(f1_scores)
    
    return final_exact, final_f1

if __name__ == '__main__':
    main()