import json

def f1_score(pred, answer):
    pcnt = 0
    for word in pred:
        if word in answer:
            pcnt += 1
    rcnt = 0    
    for word in answer:
        if word in pred:
            rcnt += 1
            
    precision = pcnt / len(pred)
    recall = rcnt / len(answer)
    # cover divide by zero exception
    if precision + recall == 0.:
        return 0.
    # harmonic mean
    f1 = (2*precision*recall)/(precision + recall)
    
    return f1

def preprocess(file_path):
    context_list = []
    context_map = {}
    question_list = []
    answer_list = []
    
    with open(file_path, "r", encoding='utf-8') as file:
        reader = json.load(file)["data"]

        ctx_i, qa_i = 0, 0
        for data in reader:
            for paragraph in data['paragraphs']:
                context_list.append(paragraph['context'])
                
                for qa in paragraph['qas']:
                    question_list.append(qa['question'])
                    answers = []
                    for answer in qa['answers']:
                        answers.append(answer['text'])
                    answer_list.append(answers)
                    context_map[str(qa_i)] = ctx_i
                    qa_i += 1
                ctx_i += 1
                
    return {
        "context_list": context_list,
        "context_map": context_map,
        "question_list": question_list,
        "answer_list": answer_list
    }