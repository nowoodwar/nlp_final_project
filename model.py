import torch
import torch.nn as nn
from transformers import BertModel

class BertBase(nn.Module):
    def __init__(self, bert_path, embedding_size, max_seq_len):
        super(BertBase, self).__init__()
        self.bert_path = bert_path
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.start = nn.Linear(embedding_size, max_seq_len)
        self.end = nn.Linear(embedding_size, max_seq_len)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, input_ids, segment_ids, mask):
        # Give 3 input id lists to BERT, ignore output[0], containing losses
        _, output = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=mask)
        bert_out = self.dropout(output)
        # Apply (start, end) weights to BERT output
        start_logits = self.start(bert_out)
        end_logits = self.end(bert_out)
        
        # Apply softmax to get prob distribution of start and end indices
        start = torch.sigmoid(start_logits)
        end = torch.sigmoid(end_logits)
        
        return start, end

class BertDataset():
    def __init__(self, q_list, context_list, a_list, context_map, tokenizer, max_len):
        self.context_list = context_list
        self.context_map = context_map
        self.q_list = q_list
        self.a_list = a_list
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.a_list)
    
    def __getitem__(self, idx):
        # Context map to save space because there are multiple Q/A's for each body
        question = self.q_list[idx]
        context = self.context_list[self.context_map[str(idx)]]
        answers = self.a_list[idx]
        start_targets, end_targets = [], []
        
        # Tokenized and formatted for BERT input
        encoding = self.tokenizer.encode_plus(
            question, 
            context, 
        )
        input_ids = encoding["input_ids"]
        segment_ids = encoding["token_type_ids"]
        mask = encoding["attention_mask"]   
        
        # Calculate the target indices within tokenized input_ids
        for answer in answers:
            start_target, end_target = 0, 0
            answer_ids = self.tokenizer.encode(answer)
            answer_ids = answer_ids[1:-1] # remove special tok ids to match accurately
            a_len = len(answer_ids)
            for i in [e for e, first in enumerate(input_ids) if first == answer_ids[0]]:
                if input_ids[i:i+a_len] == answer_ids:
                    start_target = i
                    end_target = i + a_len - 1
                    break
            start_targets.append(start_target)
            end_targets.append(end_target)
        
        # Pad id lists with 0s to self.max_len, cut at max_len, check targets not OOB
        diff = self.max_len - len(input_ids)
        if diff < 0:
            input_ids = input_ids[:self.max_len]
            segment_ids = segment_ids[:self.max_len]
            mask = mask[:self.max_len]
            for i in range(len(end_targets)):
                if end_targets[i] > 511:
                    start_targets[i] = 511
                    end_targets[i] = 511
        else:
            pad = [0]*diff
            input_ids += pad
            segment_ids += pad
            mask += pad
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "start_targets": torch.tensor(start_targets, dtype=torch.long),
            "end_targets": torch.tensor(end_targets, dtype=torch.long)
        }