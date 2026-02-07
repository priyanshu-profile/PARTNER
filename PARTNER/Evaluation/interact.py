import sys
sys.path.append('PATH TO DIRECTORY WITH RL FINE-TUNED MODEL')
from dataset import PersuadeDataset
import os, pdb
import time
import spacy
import numpy as np
from nltk.translate.meteor_score import meteor_score
import nltk
import pandas as pd
import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, RobertaForSequenceClassification, RobertaTokenizer
torch.autograd.set_detect_anomaly(True)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('bert-base-nli-mean-tokens')

import warnings
warnings.filterwarnings('ignore')

GEN = int(input("Enter the number of examples to generate:"))

def seed(seed=10):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def extract_data(self, csvfile):
    df = pd.read_csv(csvfile)
    data = []
    for i in tqdm.trange(len(df)):
        if df['authorRole'][i] == 0:
            text = "A:" + str(df["utterance"][i])
            if self.counseling_classifier and self.politeness_classifier and self.empathy_classifier:
                counseling_id = int(df['act'][i])
                politeness_id = int(df['politeness'][i])
                empathy_id = int(df['empathy'][i])
                tup = (text, counseling_id, politeness_id, empathy_id)
            elif self.counseling_classifier and self.politeness_classifier and not self.empathy_classifier:
                counseling_id = int(df['act'][i])
                politeness_id = int(df['politeness'][i])
                tup = (text, counseling_id, politeness_id)
            elif self.counseling_classifier and not self.politeness_classifier and self.empathy_classifier:
                counseling_id = int(df['act'][i])
                empathy_id = int(df['empathy'][i])
                tup = (text, counseling_id, empathy_id)
            elif self.counseling_classifier and not self.politeness_classifier and not self.empathy_classifier:
                counseling_id = int(df['act'][i])
                tup = (text, counseling_id)
            elif not self.counseling_classifier and self.politeness_classifier and self.empathy_classifier:
                politeness_id = int(df['politeness'][i])
                empathy_id = int(df['empathy'][i])                    
                tup = (text, politeness_id, empathy_id)
            elif not self.counseling_classifier and self.politeness_classifier and not self.empathy_classifier:
                politeness_id = int(df['politeness'][i])
                empathy_id = int(df['empathy'][i])
                tup = (text, politeness_id)
            elif not self.counseling_classifier and not self.politeness_classifier and self.empathy_classifier:
                empathy_id = int(df['empathy'][i])                    
                tup = (text, empathy_id)
            else:
                tup = (text)
        else:
            text = "B:" + str(df["utterance"][i])
            if self.counseling_classifier and self.politeness_classifier and self.empathy_classifier:
                counseling_id = None
                politeness_id = None
                empathy_id = None
                tup = (text, counseling_id, politeness_id, empathy_id)
            elif self.counseling_classifier and self.politeness_classifier and not self.empathy_classifier:
                counseling_id = None
                politeness_id = None
                tup = (text, counseling_id, politeness_id)
            elif self.counseling_classifier and not self.politeness_classifier and self.empathy_classifier:
                counseling_id = None
                empathy_id = None
                tup = (text, counseling_id, empathy_id)
            elif self.counseling_classifier and not self.politeness_classifier and not self.empathy_classifier:
                counseling_id = None
                tup = (text, counseling_id)
            elif not self.counseling_classifier and self.politeness_classifier and self.empathy_classifier:
                politeness_id = None
                empathy_id = None                   
                tup = (text, politeness_id, empathy_id)
            elif not self.counseling_classifier and self.politeness_classifier and not self.empathy_classifier:
                politeness_id = None
                empathy_id = None
                tup = (text, politeness_id)
            elif not self.counseling_classifier and not self.politeness_classifier and self.empathy_classifier:
                empathy_id = None                    
                tup = (text, empathy_id)
            else:
                tup = (text)
        data.append(tup)
    return data

def utteranceToConversation(self, csvfile, data):
  df = pd.read_csv(self.csvfile)
  values=df['dialogueId'].unique().tolist()
  conv_ids = df['dialogueId'].tolist()
  dataset = []
  conversation = []
  for conv in values:
    for i in range(0,df.shape[0]):
      if(conv_ids[i]==conv):
        conversation.append(data[i])
      else:
        continue
    dataset.append(conversation)
    conversation = []
    
  return dataset 


def convertDicttoList(data: dict):
    return list(data.values())


def random_split_data(self, data):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    train_data = [data[idx] for idx in indices[:800]]
    val_data = [data[idx] for idx in indices[800:]]


def convert_sentences_to_strings(sentences:list, tokenizer):
    str_sentences = []
    for i in sentences:
        str_sentences.append(tokenizer.decode(i.tolist()[0][2:-2])) # Excludeqs the zero shot tokens: {A:, B:} and the End of turn tokens: [628, 198]
    return str_sentences


def expand_inputs_for_N_candidates(inputs, num_candidates):
    # inputs = inputs[None, ...]
    return inputs.repeat((num_candidates, 1))


def modify_generated_sequence(generated_sequences):
    final_generated_sequences = []
    for i in range(generated_sequences.shape[0]):
        batch_tokens = []
        for j in range(len(generated_sequences[i])):
            if generated_sequences[i][j] != 628 and generated_sequences[i][j] != -1:
                batch_tokens.append(generated_sequences[i][j])
            elif generated_sequences[i][j] == 628:
                batch_tokens.append(generated_sequences[i][j])
                batch_tokens.append(198)
                break
            else:
                break
        final_generated_sequences.append(torch.tensor(batch_tokens).unsqueeze(0))
    
    return final_generated_sequences


def top_p_candidates(logits, prob=0.92, filter_value=-float('Inf')):
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cum_sum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cum_sum > prob
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter_(1, index=sorted_indices, src=sorted_indices_to_remove.clone())
    logits[indices_to_remove] = filter_value
    
    return logits


def generate_n_candidates(model, inputs, top_p,  temperature, num_candidates, max_gen_length, past,
                          device, eos_token_id=628, pad_token_id=198):
    curr_len = 2
    inputs = expand_inputs_for_N_candidates(inputs, num_candidates)
    inputs_ = inputs
    generated_sequences = torch.ones((inputs.shape[0], max_gen_length), dtype=torch.long) * -1
    generated_sequences[:, 0:2] = inputs.cpu()
    unfinished_sequences = inputs.new(inputs.shape[0]).fill_(1) #.cpu()
    i = 0
    while True:
        if past:
            if past[0][0].shape[-2] > 1024:
                if not torch.all(generated_sequences==-1):
                    final_generated_sequence, final_generated_log_probs = modify_generated_sequence(generated_sequences, generated_token_log_prob)
                    return final_generated_sequence, final_generated_log_probs, past_to_return
                else:
                    return None, None
        outputs = model(inputs, past, return_dict=False)
        logits, past = outputs[0], outputs[1]
        next_token_logits = logits[:, -1, :].contiguous() / temperature
        if top_p and top_p > 0.0:
            # This returns score after performing softmax function.
            next_token_logits = top_p_candidates(next_token_logits, top_p)
            next_token_log_probs = F.log_softmax(next_token_logits, -1)
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            #next_token_log_probs = next_token_log_probs.gather(-1, next_tokens)
            next_tokens = next_tokens.squeeze(1)
            if eos_token_id is not None:
                assert pad_token_id is not None # "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            generated_sequences[:, curr_len] = next_tokens.cpu()
            inputs = next_tokens.unsqueeze(1).to(device)
            curr_len = curr_len + 1
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            if unfinished_sequences.max() == 0:
                break
            if curr_len >= max_gen_length:
                break
    final_generated_sequences = modify_generated_sequence(generated_sequences)

    return final_generated_sequences


def normalize(text, nlp):
    sent = ''
    doc = nlp(text)
    for token in doc:
        if not token.is_punct:
            sent += token.lemma_
            sent += ' '
    return sent


def jaccard_similarity(context_sentence_list, generated_sentence, nlp):
    str1 = context_sentence_list[0]
    str1 = normalize(str1, nlp)
    str1 = set(str1.split())
    jacc_score = []
    for i in generated_sentence:
        str2 = i
        str2 = normalize(str2, nlp)
        str2 = set(str2.split())
        sim_score = float(len(str1 & str2)) / len(str1 | str2)
        jacc_score.append(sim_score)
    return jacc_score


def filter_response(generated_responses, jaccard_score):
    
    filtered_responses = []
    for i in range(len(generated_responses)):
        if jaccard_score[i] >= 0.5:
            #continue
            filtered_responses.append(generated_responses[i])
        else:
            continue
    return filtered_responses

def response_with_act(binary_classifier, binary_tokenizer, generated_responses, device, get_best):
    
    inputs = binary_tokenizer(generated_responses, padding=True, return_tensors='pt')
    outputs = binary_classifier(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))
    probs = F.softmax(outputs.logits, dim=-1)
    _, pred_label = torch.topk(probs, k=1)
    num_act = pred_label.sum().item() / num_candidates

    if not get_best:
        return num_act
    try:
        index_containing_act = list(np.where(np.array(pred_label.cpu()) != 0)[0])
    except:
        pass
    if len(index_containing_act) > 1:
        # Select the one with bigger length:
        lengths = [len(generated_responses[i].split()) for i in index_containing_act]
        return generated_responses[np.argmax(lengths)]
    elif len(index_containing_act) == 1:
        return generated_responses[index_containing_act[0]]
    else:
        lengths = [len(i.split()) for i in generated_responses]
        return generated_responses[np.argmax(lengths)]

def get_best_candidate(generated_sequences, binary_classifier, tokenizer, binary_tokenizer,
                       device, context_sentence_list, nlp, get_best):
    generated_responses = convert_sentences_to_strings(generated_sequences, tokenizer)
    if get_best:
        '''if len(context_sentence_list) != 0:
            jaccard_score = jaccard_similarity(context_sentence_list, generated_responses, nlp)
            filtered_responses = filter_response(generated_responses, jaccard_score)
        else:
            filtered_responses = generated_responses'''
        best_response = response_with_act(binary_classifier, binary_tokenizer, filtered_responses, device, get_best)
        return best_response
    else:
        num_response_with_act =  response_with_act(binary_classifier, binary_tokenizer, generated_responses, device, get_best)
        # print(num_response_with_act)
        return num_response_with_act


csvfile = 'PATH TO ANNOTATED CSV FILE'
get_best = False # MAKE THIS TRUE TO GET THE BEST RL CANDIDATE

seed()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
batch_size = 1

data = extract_data(csvfile=csvfile, Counseling_classifier=False, politeness_classifier=None)
data = utteranceToConversation(csvfile, data)
traindata, valdata = random_split_data(data)
valdata = PersuadeDataset(valdata, tokenizer)
val_dataloader = DataLoader(dataset=valdata,
                            shuffle=False,
                            batch_size=batch_size,
                            collate_fn=valdata.collate)

nlp = spacy.load("en_core_web_sm")

RL_model_A = GPT2LMHeadModel.from_pretrained("gpt2")

device = 'cuda' #torch.device("cuda")

RL_model_A = RL_model_A.to(device)

RL_model_A_state_dict = torch.load("PATH TO SAVED RL FINE-TUNED MODEL")

RL_model_A.load_state_dict(RL_model_A_state_dict)

RL_model_A.eval()

## Loading the Persuasasion Binary Classifier for selecting the best out of the generated candidates:
con_classifier_filename = "PATH TO SAVED COUNSELING ACT CLASSIFIER MODEL"
Counseling_num_labels = 2 ## Binary classifier
con_model_dict = torch.load(con_classifier_filename)
binary_classifier_1 = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=Counseling_num_labels)
binary_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
binary_classifier_1.config.problem_type = 'single_label_classification'
binary_classifier_1.load_state_dict(con_model_dict['state_dict'])
binary_classifier_1 = binary_classifier_1.to(device)
binary_classifier_1.eval()

## Loading the Politeness Binary Classifier for selecting the best out of the generated candidates:
pol_classifier_filename = "PATH TO POLITENESS CLASSIFIER MODEL"
politeness_num_labels = 2 
pol_model_dict = torch.load(pol_classifier_filename)
binary_classifier_2 = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=politeness_num_labels)
binary_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
binary_classifier_2.config.problem_type = 'single_label_classification'
binary_classifier_2.load_state_dict(pol_model_dict['state_dict'])
binary_classifier_2 = binary_classifier_2.to(device)
binary_classifier_2.eval()

## Loading the Empathy Binary Classifier for selecting the best out of the generated candidates:
emp_classifier_filename = "PATH TO EMPATHY CLASSIFIER MODEL"
empathy_num_labels = 2 
emp_model_dict = torch.load(emp_classifier_filename)
binary_classifier_3 = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=empathy_num_labels)
binary_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
binary_classifier_3.config.problem_type = 'single_label_classification'
binary_classifier_3.load_state_dict(emp_model_dict['state_dict'])
binary_classifier_3 = binary_classifier_2.to(device)
binary_classifier_3.eval()



prev_input = tokenizer.encode('A:')
prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(device)

temperature = 0.8
top_k = 400
top_p = 0.9
max_length = 100
num_candidates = 3

RL_past = None

sep = [628, 198]


ground_truth = []
RL_candidates = []

B4 = []
ID = []

id_ = 0
count = 0

number_of_responses_with_con_act = 0
number_of_responses_with_politeness = 0
number_of_responses_with_empathy = 0

with torch.no_grad():
    
    for idx, batch in enumerate(val_dataloader):
        
        RL_past = None
        
        if sum([len(item) for item in batch[0][1]]) > 1024:
            continue
        
        role_ids, dialog_tokens = batch[0]
        context_sentence_list = []
        
        dial_inputs = [torch.LongTensor(item).unsqueeze(0) for item in dialog_tokens]

        candidate_sentences = []


        lengths = []

        for num_turn, dialog_turn_inputs in enumerate(dial_inputs):
            
            assert not np.any(np.isnan(dialog_turn_inputs).cpu().numpy()), 'Inputs Dialog contains Nan value.'
            
            dialog_turn_inputs = dialog_turn_inputs.to(device)
            
            if role_ids[num_turn] == 0:
                generated_sequences = generate_n_candidates(RL_model_A, torch.tensor(tokenizer.encode("A:")).unsqueeze(0).to(device),
                                                            top_p=top_p,
                                                            temperature=temperature,
                                                            num_candidates=num_candidates,
                                                            max_gen_length=max_length,
                                                            past=RL_past,
                                                            device=device,
                                                            eos_token_id=628,
                                                            pad_token_id=198)

                for i in generated_sequences:
                    candidate = tokenizer.decode(i.tolist()[0][2:])
                    candidate_sentences.append(candidate.split('\t')[0])

                num_con_act = get_best_candidate(generated_sequences=generated_sequences,
                                                  binary_classifier=binary_classifier_1,
                                                  tokenizer=tokenizer,
                                                  binary_tokenizer=binary_tokenizer,
                                                  device=device,
                                                  context_sentence_list=context_sentence_list,
                                                  nlp=nlp,
                                                  get_best=get_best)
                number_of_responses_with_con_act += num_con_act
                
                num_pol = get_best_candidate(generated_sequences=generated_sequences,
                                                  binary_classifier=binary_classifier_2,
                                                  tokenizer=tokenizer,
                                                  binary_tokenizer=binary_tokenizer,
                                                  device=device,
                                                  context_sentence_list=context_sentence_list,
                                                  nlp=nlp,
                                                  get_best=get_best)
                number_of_responses_with_politeness += num_pol

                num_emp = get_best_candidate(generated_sequences=generated_sequences,
                                                  binary_classifier=binary_classifier_3,
                                                  tokenizer=tokenizer,
                                                  binary_tokenizer=binary_tokenizer,
                                                  device=device,
                                                  context_sentence_list=context_sentence_list,
                                                  nlp=nlp,
                                                  get_best=get_best)
                number_of_responses_with_empathy += num_emp



                _, RL_past = RL_model_A(expand_inputs_for_N_candidates(dialog_turn_inputs, num_candidates), RL_past, return_dict=False)

                
                ground_truth_string = convert_sentences_to_strings([dialog_turn_inputs], tokenizer)[0]

                for i in range(len(candidate_sentences)):
                    lengths.append(len(candidate_sentences[i].split()))


                ground_truth.append(ground_truth_string)
                B4.append(0)
                ID.append(id_)
                
                if len(context_sentence_list) == 0:
                    context_sentence_list.append(ground_truth_string)
                else:
                    context_sentence_list.pop()
                    context_sentence_list.append(ground_truth_string)

                count +=1
            else:
                _, RL_past = RL_model_A(expand_inputs_for_N_candidates(dialog_turn_inputs, num_candidates), RL_past, return_dict=False)

                
                RL_candidates.append(None)
                
                ground_truth_string = convert_sentences_to_strings([dialog_turn_inputs], tokenizer)[0]
                ground_truth.append(ground_truth_string)
                B4.append(1)
                ID.append(id_)
                
        id_ += 1

        if count >= GEN:
            break
            
            
    print(f"Percentage of Utterances with Counseling act {number_of_responses_with_con_act/count}")
    print(f"Percentage of Utterances with Politeness {number_of_responses_with_politeness/count}")
    print(f"Percentage of Utterances with Empathy {number_of_responses_with_empathy/count}")
    print(f"Average Length of Candidates {np.mean(np.array(lengths))}")