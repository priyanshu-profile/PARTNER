import numpy as np
import torch.nn as nn
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import functools
import operator
import pdb
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import warnings
import re
warnings.filterwarnings("ignore")

model = SentenceTransformer('bert-base-nli-mean-tokens')

def convert_sentences_to_strings(sentences:list, tokenizer):
    str_sentences = []
    for i in sentences:
        str_sentences.append(tokenizer.decode(i.tolist()[0][2:-2])) # Excludeqs the zero shot tokens: {A:, B:} and the End of turn tokens: [628, 198]
    return str_sentences

def normalize(text, nlp):
    sent = ''
    doc = nlp(text)
    for token in doc:
        if not token.is_punct:
            sent += token.lemma_
            sent += ' '
    return sent

def calculate_naturalness_reward(candidates, tokenizer, model):
    """
    candidates: list of generated token tensors (batch of [1, seq_len] each)
    tokenizer: the tokenizer associated with the language model
    model: autoregressive LM (e.g., GPT2, DialoGPT) in eval mode, returns loss/logits
    """
    naturalness_scores = []

    model.eval()
    for candidate in candidates:
        input_ids = candidate.squeeze(0)
        
        # Shift labels so model is predicting next token
        inputs = input_ids.unsqueeze(0).to(model.device)
        with torch.no_grad():
            outputs = model(inputs, labels=inputs)
            nll = outputs.loss.item()  # scalar

        # Apply scaled tanh
        reward = -np.tanh(nll)
        naturalness_scores.append(reward)

    return naturalness_scores

# Helper to get grounded tokens (e.g., lowercased non-stopword alphanumerics)
def extract_grounded_tokens(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return set([t for t in tokens if t not in ENGLISH_STOP_WORDS])

def calculate_retaiment(candidates, current_sentence, tokenizer, num_turn, dial_inputs):
    retainment_reward_scores = []

    for cand, prev in zip(candidates, previous_candidates):
        # Decode both candidate and previous candidate
        cand_text = tokenizer.decode(cand.tolist()[0][2:]).split('\n')[0]
        prev_text = tokenizer.decode(prev.tolist()[0][2:]).split('\n')[0]

        # Cosine similarity between embeddings
        embeddings = model.encode([cand_text, prev_text])
        cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        # Jaccard similarity between grounded token sets
        set1 = extract_grounded_tokens(cand_text)
        set2 = extract_grounded_tokens(prev_text)
        jaccard_sim = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0.0

        reward = 0.5 * (cos_sim + jaccard_sim)
        retainment_reward_scores.append(reward)

    return retainment_reward_scores

def get_counseling_reward(generated_sentences,
                          counseling_classifier,
                          actual_counseling_label,
                          device,
                          tokenizer,
                          counseling_tokenizer,
                          beta2):
    
    gen_str_sent = convert_sentences_to_strings(generated_sentences, tokenizer)
    
    inputs = counseling_tokenizer(gen_str_sent, return_tensors='pt', padding=True, truncation=True)
    
    output = counseling_classifier(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))
    
    probs = F.softmax(output.logits, dim=-1)
    actual_label_prob = probs[:, actual_counseling_label]
    other_label_prob = probs.sum(-1) - actual_label_prob
    
    reward_dict = {'actual_prob': actual_label_prob.tolist(),
                   'other_prob': other_label_prob.tolist()}
    
    reward = actual_label_prob - beta2 * other_label_prob
    
    return reward_dict, reward.tolist()

def get_politeness_reward(generated_sentences,
                          politeness_classifier,
                          actual_politeness_label,
                          device,
                          tokenizer, 
                          politeness_tokenizer,
                          beta3):

    gen_str_sent = convert_sentences_to_strings(generated_sentences, tokenizer)
    
    inputs = politeness_tokenizer(gen_str_sent, return_tensors='pt', padding=True, truncation=True)
    
    output = politeness_classifier(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))
    
    probs = F.softmax(output.logits, dim=-1)
    actual_label_prob = probs[:, actual_politeness_label]
    
    other_label_prob = probs.sum(-1) - actual_label_prob

    reward_dict = {'actual_prob': actual_label_prob.tolist(),
                   'other_prob': other_label_prob.tolist()}
    
    reward = actual_label_prob - beta3 * other_label_prob

    return reward_dict, reward.tolist()


def get_empathy_reward(generated_sentences,
                       empathy_classifier
                       actual_empathy_label,
                       device,
                       tokenizer, 
                       empathy_tokenizer,
                       beta4):

    gen_str_sent = convert_sentences_to_strings(generated_sentences, tokenizer)
    
    inputs = empathy_tokenizer(gen_str_sent, return_tensors='pt', padding=True, truncation=True)
    
    output = empathy_classifier(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))
    
    probs = F.softmax(output.logits, dim=-1)
    actual_label_prob = probs[:, actual_politeness_label]
    
    other_label_prob = probs.sum(-1) - actual_label_prob

    reward_dict = {'actual_prob': actual_label_prob.tolist(),
                   'other_prob': other_label_prob.tolist()}
    
    reward = actual_label_prob - beta4 * other_label_prob

    return reward_dict, reward.tolist()

def calculate_rewards(model_A,
                      current_sentence,
                      num_turn,
                      dial_inputs,
                      length,
                      generated_sentences,
                      source_list,
                      tokenizer,
                      criterion,
                      use_naturalness,
                      use_context,
                      nlp,
                      device,
                      beta2,
                      beta3,
                      beta4,
                      counseling_tokenizer,
                      gamma1,
                      gamma2,
                      gamma3,
                      gamma4,
                      counseling_classifier,
                      actual_counseling_label,
                      counselor=False,
                      gamma5=None,
                      politeness_classifier=None,
                      actual_politeness_label=None,
                      empathy_classifier=None,
                      empathy_politeness_label=None):
    
    scores = {}

    scores['cossim'] = []
    scores['context'] = []
    scores['naturalness'] = []
    scores['counseling'] = []
    scores['politeness'] = []
    scores['empathy'] = []
    scores['counseling_actual_prob'] = []
    scores['counseling_other_prob'] = []
    scores['politeness_actual_prob'] = []
    scores['politeness_other_prob'] = []
    scores['empathy_actual_prob'] = []
    scores['empathy_other_prob'] = []

    if len(generated_sentences) >= 1:
        
        rewards = np.zeros((len(generated_sentences)))
        
        # if (len(source_list) ==2):
            
        if counseling_classifier:   # generated_sentences, counseling_classifier, actual_counseling_label, device, tokenizer, counseling_tokenizer, beta2
            counseling_dict, counseling_probs = get_counseling_reward(generated_sentences,
                                                                      counseling_classifier,
                                                                      actual_counseling_label,
                                                                      device,
                                                                      tokenizer,
                                                                      counseling_tokenizer,
                                                                      beta2)
            rewards += gamma1* np.array(counseling_probs)
        
        if not counseling_classifier:
            counseling_dict =  {'actual_prob': [None], 'other_prob': [None]}
            counseling_probs = [None]
        
        if politeness_classifier:
            politeness_dict, politeness_probs = get_politeness_reward(generated_sentences,
                                                                      politeness_classifier,
                                                                      actual_politeness_label,
                                                                      device, tokenizer, 
                                                                      counseling_tokenizer,
                                                                      beta3)
            rewards += gamma2* np.array(politeness_probs)

        if not politeness_classifier:
            politeness_dict =  {'actual_prob': [None], 'other_prob': [None]}
            politeness_probs = [None]

        if empathy_classifier:
            empathy_dict, empathy_probs = get_empathy_reward(generated_sentences,
                                                                    empathy_classifier,
                                                                    actual_empathy_label,
                                                                    device, tokenizer, 
                                                                    counseling_tokenizer,
                                                                    beta4)
            rewards += gamma3* np.array(empathy_probs)

        if not empathy_classifier:
            empathy_dict =  {'actual_prob': [None], 'other_prob': [None]}
            empathy_probs = [None]



        if use_naturalness: # use naturalness
            non_rep = calculate_naturalness_reward(source_list, generated_sentences, tokenizer, nlp)
            
            non_rep = np.array(non_rep)

            rewards -= gamma4*(non_rep)
        else: 
            non_rep = None
        

        if use_context:
            retainment_scores = calculate_retaiment(generated_sentences, current_sentence, tokenizer, num_turn, dial_inputs)
            rewards += gamma5*np.array(context_adequacy_scores)
        
        if not use_context:
            retainment_scores = None

    else:
        rewards = 0
        naturalness_dist = calculate_naturalness_reward(current_sentence, generated_sentences, tokenizer, nlp)
        
        rewards -= naturalness_dist
    try:
        scores['naturalness'].extend(naturalness_dist)
    except:
        pass
    
    scores['naturalness'].extend(engagingness.tolist())
    scores['context'].extend(retainment_scores)
    scores['counseling'].extend(counseling_probs)
    scores['politeness'].extend(politeness_probs)
    scores['empathy'].extend(empathy_probs)
    scores['counseling_actual_prob'].extend(counseling_dict['actual_prob'])
    scores['counseling_other_prob'].extend(counseling_dict['other_prob'])
    scores['politeness_actual_prob'].extend(politeness_dict['actual_prob'])
    scores['politeness_other_prob'].extend(politeness_dict['other_prob'])
    scores['empathy_actual_prob'].extend(empathy_dict['actual_prob'])
    scores['empathy_other_prob'].extend(empathy_dict['other_prob'])


    return list(rewards), scores

def append(generated_list, context_sentence, tokenizer):
    
    if len(generated_list) == 2:
        generated_list.pop(0)
        cntx = tokenizer.decode(context_sentence.tolist()[0][2:]).split('\n')[0]
        generated_list.append(cntx)
    else:
        cntx = tokenizer.decode(context_sentence.tolist()[0][2:]).split('\n')[0]
        generated_list.append(cntx)
    
    return generated_list

def expand_inputs_for_N_candidates(inputs, num_candidates):
    # inputs = inputs[None, ...]
    return inputs.repeat((num_candidates, 1))

def modify_generated_sequence(generated_sequences, generated_log_probs):
    
    final_generated_sequences = []
    final_generated_log_probs = []
    
    for i in range(generated_sequences.shape[0]):
        
        batch_tokens = []
        batch_log_probs = []
        
        for j in range(len(generated_sequences[i])):
            if generated_sequences[i][j] != 628 and generated_sequences[i][j] != -1:
                batch_tokens.append(generated_sequences[i][j])
                batch_log_probs.append(generated_log_probs[i][j])
            elif generated_sequences[i][j] == 628:
                batch_tokens.append(generated_sequences[i][j])
                batch_log_probs.append(generated_log_probs[i][j])
                batch_tokens.append(198)
                break
            else:
                break
        final_generated_sequences.append(torch.tensor(batch_tokens).unsqueeze(0))
        ### BE CAREFUL WHEN USING THIS, SINCE IT DOESN NOT AVERAGES THE LOG PROBS INSTEAD IT JUST TAKES THE SUM.
        final_generated_log_probs.append(torch.tensor(batch_log_probs).sum().item())
    
    return final_generated_sequences, final_generated_log_probs

def top_p_candidates(logits, prob=0.92, filter_value=-float('Inf')):
    
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cum_sum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cum_sum > prob
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter_(1, index=sorted_indices, src=sorted_indices_to_remove.clone())
    logits[indices_to_remove] = filter_value
    
    return logits

def generate_n_candidates(model,
                          inputs,
                          top_p,
                          temperature,
                          num_candidates,
                          max_gen_length,
                          past,
                          device,
                          eos_token_id=628,
                          pad_token_id=198):

    curr_len = 2

    inputs = expand_inputs_for_N_candidates(inputs, num_candidates)
    inputs_ = inputs
    
    generated_sequences = torch.ones((inputs.shape[0], max_gen_length), dtype=torch.long) * -1
    generated_sequences[:, 0:2] = inputs.cpu()
    
    generated_token_log_prob = torch.zeros((inputs.shape[0], max_gen_length), dtype=torch.float)
    
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
            next_token_log_probs = next_token_log_probs.gather(-1, next_tokens)
            next_tokens = next_tokens.squeeze(1)
            
            if eos_token_id is not None:
                assert pad_token_id is not None # "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # NOTE: SAVE LOG PROBS AS WELL
            generated_sequences[:, curr_len] = next_tokens.cpu()
            inputs = next_tokens.unsqueeze(1).to(device)
            #inputs_ = torch.cat((inputs_, next_tokens[:, None]), dim=-1)
            
            curr_len = curr_len + 1
            
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            if unfinished_sequences.max() == 0:
                break
            if curr_len >= max_gen_length:
                break
    
    final_generated_sequences, final_generated_log_probs =  modify_generated_sequence(generated_sequences, generated_token_log_prob)
    
    return final_generated_sequences, final_generated_log_probs

def compute_log_probs(target_token_ids, logits, mask, average_sent_loss=False):
    logits = logits[:, :-1, :].contiguous() # (batch, sequence_length, vocab_size)
    
    target_token_ids = target_token_ids[:, 1:].contiguous() # (batch, sequence_length)
    

    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, -1, target_token_ids.unsqueeze(-1)).squeeze(-1)
    mask = mask[:, 1:].contiguous()
    
    if average_sent_loss:
        log_probs = (log_probs * mask).sum(-1) / mask.sum(-1)
    else:
        log_probs = (log_probs * mask).sum(-1)
    return {'log_probs': log_probs}

def ppo_step(model_A,
             model_B,
             buffer_memory,
             device,
             ppo_epsilon,
             num_candidates,
             criterion,
             optimizer,
             dial_inputs,
             role_ids,
             scheduler=None,
             train_single_model=False,
             model_to_train=None,
             average_sent_loss=False,
             use_recent_past=False):

    optimizer.zero_grad()
    
    log_dict = {}
    
    new_log_prob = []
    old_log_prob = []
    
    rewardlist = []
    
    ratios = []
    
    policy_loss = []
    advantages  = []

    if use_recent_past:
        print('USING RECENT PAST')
    else:
        print('NOT USING RECENT PAST')

    if use_recent_past:
        
        batches = buffer_memory.get_batch(shuffle=False)
        
        past = None
        
        i = 1
        
        for idx, batch in enumerate(batches):
            
            action = torch.tensor(batch['action'], device=device).unsqueeze(0)
            #pdb.set_trace()      
            if batch['human_response']:
                
                if idx == 0:
                    logits, past = model_A(action, past, return_dict=False)
                
                if idx > 0 and idx % (num_candidates + 1) == 0:
                    try:
                        past = out
                    except:
                        pass
                    
                    #history_indices = idx // (num_candidates + 1)
                    #history = dial_inputs[history_indices]
                    
                    history = dial_inputs[i]
                    
                    _, past = model_A(history.to(device), past_key_values=past, return_dict=False)
                    logits, out = model_A(action, past_key_values=past, return_dict=False)
                    
                    i += 2
            else:
                history_indices = idx // (num_candidates + 1)  # {A:(1,2,3,4,5),B, C:(7,8,9,10,11), D, E: (13,14,15,16,17)}
                
                if history_indices == 0:
                    logits, _ = model_A(action, past_key_values=None, return_dict=False)
                else:
                    logits, _ = model_A(action, past_key_values=past, return_dict=False)
            
            new_log_probs = compute_log_probs(target_token_ids=action,
                                              logits=logits,
                                              mask=torch.ones_like(action).to(device),
                                              average_sent_loss=average_sent_loss)['log_probs']

            old_log_probs = torch.tensor(batch['log_prob'], device=device).unsqueeze(0)
            old_log_prob.append(old_log_probs)

            rewards = torch.tensor(batch['reward'], device=device).unsqueeze(0)
            rewardlist.append(batch['reward'])
            advantages.append(rewards)

            new_log_prob.append(new_log_probs)

        if new_log_prob:
            new_log_prob = torch.cat(new_log_prob, dim=-1)
            old_log_prob = torch.cat(old_log_prob, dim=-1)
        
            advantages = torch.cat(advantages, dim=-1)
        
            ratio = (new_log_prob - old_log_prob).exp()
        
            policyloss1 = - advantages * ratio
            policyloss2 = - advantages * ratio.clamp(1 - ppo_epsilon, 1 + ppo_epsilon)
        
            policyloss = torch.min(policyloss1, policyloss2).mean()
        
            policyloss.backward()

            with torch.no_grad():
                log_dict['policy_loss'] = policyloss.item()
                print('Policy Loss: ', log_dict['policy_loss'])
                
                # (r-1) - logr, where r = p(x)/q(x); p(x) = new distribution and q(x) is old distribution
                log_dict['approx_kl'] = torch.mean(((new_log_prob - old_log_prob).exp() - 1)\
                                                - (new_log_prob - old_log_prob)).item()
                #log_dict['approx_kl'] = 0.5 * np.mean(np.power((np.array(new_log_prob) - np.array(old_log_prob)), 2))
                print('approx KL div: ', log_dict['approx_kl'])
                
                log_dict['clip_frac'] = torch.mean((torch.abs(ratio-1) > ppo_epsilon).float()).item()
                print('clip frac: ', log_dict['clip_frac'])
                
                log_dict['reward'] = np.mean(rewardlist)
                print('rewards: ', log_dict['reward'])
        else:
            log_dict['policy_loss'] = 0
            print('Policy Loss: ', log_dict['policy_loss'])
                
            # (r-1) - logr, where r = p(x)/q(x); p(x) = new distribution and q(x) is old distribution
            log_dict['approx_kl'] = 0
            
            #log_dict['approx_kl'] = 0.5 * np.mean(np.power((np.array(new_log_prob) - np.array(old_log_prob)), 2))
            print('approx KL div: ', log_dict['approx_kl']) 

            log_dict['clip_frac'] = 0
            print('clip frac: ', log_dict['clip_frac'])
                
            log_dict['reward'] = 0
            print('rewards: ', log_dict['reward'])
        

    if not train_single_model:
        nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)
        nn.utils.clip_grad_norm_(model_B.parameters(), 1.0)
    else:
        if model_to_train =='counselor':
            nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)

    optimizer.step()
    #scheduler.step()

    return log_dict


@torch.no_grad()
def collect_samples(batch,
                    model_A,
                    model_B,
                    top_p,
                    eos_token_id,
                    pad_token_id,
                    max_gen_length,
                    num_candidates,
                    human_reward,
                    use_context,
                    use_naturalness,
                    buffer_memory,
                    device,
                    tokenizer,
                    criterion,
                    temperature,
                    use_recent_past,
                    average_sent_loss,
                    nlp,
                    beta2,
                    beta3,
                    beta4,
                    counseling_classifier,
                    gamma1,
                    gamma2,
                    gamma3,
                    gamma4,
                    gamma5,
                    counseling_tokenizer,
                    train_single_model=False,
                    model_to_train=None,
                    recompute_log_prob=True,
                    politeness_classifier=None,
                    empathy_classifier=None,
                    fp16=False):

    scores_dict = {}

    scores_dict['retainment_scores'] = []
    scores_dict['naturalness_scoress'] = []
    
    scores_dict['counseling_actual_prob'] = []
    scores_dict['counseling_other_prob'] = []
    scores_dict['counseling_scores'] = []
    
    scores_dict['politeness_scores'] = []
    scores_dict['politeness_actual_prob'] = []
    scores_dict['politeness_other_prob'] = []

    scores_dict['empathy_scores'] = []
    scores_dict['empathy_actual_prob'] = []
    scores_dict['empathy_other_prob'] = []

    
    if not self.counseling_classifier and not self.politeness_classifier and self.empathy_classifier:
        role_ids, dialog_tokens, empathy_label = batch
    elif not self.counseling_classifier and self.politeness_classifier and not self.empathy_classifier:
        role_ids, dialog_tokens, politeness_label = batch
    elif not self.counseling_classifier and self.politeness_classifier and self.empathy_classifier:
        role_ids, dialog_tokens, politeness_label, empathy_label = batch
    elif self.counseling_classifier and not self.politeness_classifier and not self.empathy_classifier:
        role_ids, dialog_tokens, counseling_label  = batch
    elif self.counseling_classifier and not self.politeness_classifier and self.empathy_classifier:
        role_ids, dialog_tokens, counseling_label, empathy_label  = batch
    elif self.counseling_classifier and self.politeness_classifier and not self.empathy_classifier:
        role_ids, dialog_tokens, counseling_label, politeness_label = batch
    elif self.counseling_classifier and self.politeness_classifier and self.empathy_classifier:
        role_ids, dialog_tokens, counseling_label, politeness_label, empathy_label  = batch
    else:
    	role_ids, dialog_tokens = batch


    dial_inputs = [torch.LongTensor(item).unsqueeze(0) for item in dialog_tokens]

    past = None
    past_ = None
    
    context = None
    cntxt = None

    counselor_generated_list, persuadee_generated_list = [], []
    length = np.zeros(num_candidates)
    length = length.tolist()

    for num_turn, dialog_turn_inputs in enumerate(dial_inputs):
        
        assert not np.any(np.isnan(dialog_turn_inputs).cpu().numpy()), 'Inputs Dialog contains Nan value.'
        
        dialog_turn_inputs = dialog_turn_inputs.to(device)

        if counseling_classifier and not politeness_classifier and not empathy_classifier:
            actual_counseling_label = counseling_labels[num_turn]
            actual_politeness_label = None
            actual_empathy_label = None
        elif counseling_classifier and politeness_classifier and empathy_classifier:
            actual_counseling_label = counseling_label[num_turn]
            actual_politeness_label = politeness_label[num_turn]
            actual_empathy_label = empathy_labels[num_turn]
        elif counseling_classifier and politeness_classifier and not empathy_classifier:
            actual_counseling_label = counseling_label[num_turn]
            actual_politeness_label = politeness_label[num_turn]
            actual_empathy_label = None
        elif not counseling_classifier and politeness_classifier and not empathy_classifier:
            actual_counseling_label = None
            actual_politeness_label = politeness_label[num_turn]
            actual_empathy_label = empathy_labels[num_turn]
        elif counseling_classifier and not politeness_classifier and empathy_classifier:
            actual_counseling_label = counseling_label[num_turn]
            actual_politeness_label = None
            actual_empathy_label = empathy_labels[num_turn]
        elif not counseling_classifier and politeness_classifier and not empathy_classifier:
            actual_counseling_label = None
            actual_politeness_label = politeness_label[num_turn]
            actual_empathy_label = None
        elif not counseling_classifier and not politeness_classifier and empathy_classifier:
            # politeness only
            actual_counseling_label = None
            actual_politeness_label = None
            actual_empathy_label = empathy_labels[num_turn]
        else:
            actual_counseling_label = None
            actual_politeness_label = None
            actual_empathy_label = None

        if model_to_train == 'counselor':

            if role_ids[num_turn] == 0:
                
                '''if use_recent_past:
                    if cntxt is not None:
                        past = prepare_inputs(cntxt, model_A)
                    else:
                        past = None'''
                
                #dial_turn_str = convert_sentences_to_strings([dialog_turn_inputs], tokenizer)[0]

                outputs = model_A(dialog_turn_inputs, past, return_dict=False)
                logits = outputs[0]

                mask = torch.ones_like(dialog_turn_inputs).to(device)
                
                log_probs = compute_log_probs(target_token_ids=dialog_turn_inputs,
                                              logits=logits,
                                              mask=mask,
                                              average_sent_loss=average_sent_loss)
                
                buffer_memory.update_buffer(state=dialog_turn_inputs.tolist()[0],
                                            context=context,
                                            action=dialog_turn_inputs.tolist()[0],
                                            action_log_probs=log_probs['log_probs'].item(),
                                            reward=human_reward,
                                            counselor=True,
                                            human_response=True)
                if not use_recent_past:
                    '''In this case, first we generate sentence using the entire past. And then we update the past with
                    the current utterance.'''
                    generated_sequence, generated_log_probs  = generate_n_candidates(model_A,
                                                                                     torch.tensor(tokenizer.encode("A:")).unsqueeze(0).to(device),
                                                                                     top_p,
                                                                                     eos_token_id=eos_token_id,
                                                                                     pad_token_id=pad_token_id,
                                                                                     num_candidates=num_candidates,
                                                                                     max_gen_length=max_gen_length,
                                                                                     temperature=temperature,
                                                                                     past=past_,
                                                                                     device=device)
                    output = model_A(expand_inputs_for_N_candidates(dialog_turn_inputs,num_candidates),
                                     past_,
                                     return_dict=False)

                    past_ = output[1]
                
                else:
                    '''Here first we calculate the past based on the context sentence and then we generate candidates.'''
                    '''if cntxt is not None:
                        past_ = prepare_inputs(expand_inputs_for_N_candidates(cntxt, num_candidates), model_A)
                    else:
                        past_ = None'''

                    generated_sequence, generated_log_probs = generate_n_candidates(model_A,
                                                                                    torch.tensor(tokenizer.encode("A:")).unsqueeze(0).to(device), top_p,
                                                                                    eos_token_id=eos_token_id,
                                                                                    pad_token_id=pad_token_id,
                                                                                    num_candidates=num_candidates,
                                                                                    max_gen_length=max_gen_length,
                                                                                    temperature=temperature,
                                                                                    past=past_,
                                                                                    device=device)

                #gen_sent = convert_sentences_to_strings(generated_sequence, tokenizer)
                counselor_generated_list = append(counselor_generated_list, dialog_turn_inputs, tokenizer)

                current_sentence = tokenizer.decode(dialog_turn_inputs.tolist()[0][2:]).split('\t')[0]
                if num_turn+1<len(dial_inputs):
                    next_sentence = tokenizer.decode(dial_inputs[num_turn+1].tolist()[0][2:]).split('\t')[0]
                else:
                    next_sentence = ''

                # calculation of dialogue length:
                length = dialogue_length(generated_sequence, tokenizer, length, num_turn, dial_inputs)

                reward, scores = calculate_rewards(current_sentence=current_sentence,
                                                   next_sentence = next_sentence,
                                                   num_turn=num_turn,
                                                   dial_inputs=dial_inputs,
                                                   generated_sentences= generated_sequence,
                                                   length=length,
                                                   actual_counseling_label=actual_counseling_label,
                                                   actual_politeness_label=actual_politeness_label,
                                                   actual_empathy_label=actual_empathy_label,
                                                   source_list=counselor_generated_list,
                                                   tokenizer=tokenizer,
                                                   criterion=criterion,
                                                   counseling_tokenizer=counseling_tokenizer,
                                                   counseling_classifier=counseling_classifier,
                                                   politeness_classifier=politeness_classifier,
                                                   empathy_classifier=empathy_classifier,
                                                   counselor=True,
                                                   use_naturalness=use_naturalness,
                                                   use_context=use_context,
                                                   nlp=nlp,
                                                   device=device,
                                                   beta2=beta2,
                                                   beta3=beta3,
                                                   beta4=beta4,
                                                   gamma1=gamma1,
                                                   gamma2=gamma2,
                                                   gamma3=gamma3,
                                                   gamma4=gamma4,
                                                   gamma5=gamma5,
                                                   model_A=model_A)

                #candidate_dict[dial_turn_str] = convert_sentences_to_strings(generated_sequence, tokenizer)


                scores_dict['retainment_scores'].extend(scores['context'])
                scores_dict['naturalness_scoress'].extend(scores['naturalness'])
                
                scores_dict['counseling_actual_prob'].extend(scores['counseling_actual_prob'])
                scores_dict['counseling_other_prob'].extend(scores['counseling_other_prob'])
                scores_dict['counseling_scores'].extend(scores['counseling'])

                scores_dict['politeness_actual_prob'].extend(scores['politeness_actual_prob'])
                scores_dict['politeness_other_prob'].extend(scores['politeness_other_prob'])
                scores_dict['politeness_scores'].extend(scores['politeness'])

                scores_dict['empathy_actual_prob'].extend(scores['empathy_actual_prob'])
                scores_dict['empathy_other_prob'].extend(scores['empathy_other_prob'])
                scores_dict['empathy_scores'].extend(scores['empathy'])

                if recompute_log_prob:

                    for i in range(len(generated_sequence)):
                        
                        # NOTE: STILL USING THE PAST FROM PREVIOUS UTTERANCE, SINCE WE DO NOT NEED PAST FROM
                        #       CONTAINING CURRENT UTTERANCE for GENERATED CANDIDATES
                        
                        output = model_A(generated_sequence[i].to(device), past_key_values=past, return_dict=False)
                        logits = output[0]
                        
                        log_probs = compute_log_probs(target_token_ids=generated_sequence[i].to(device),
                                                      logits=logits,
                                                      mask=torch.ones_like(generated_sequence[i]).to(device),
                                                      average_sent_loss=average_sent_loss)['log_probs'].item()
                        
                        buffer_memory.update_buffer(state=dialog_turn_inputs.tolist()[0],
                                                    context=context,
                                                    action= generated_sequence[i].tolist()[0],
                                                    action_log_probs=log_probs,
                                                    reward=reward[i],
                                                    counselor=True,
                                                    human_response=False)
                else:
                    for i in range(len(generated_sequence)):
                        buffer_memory.update_buffer(state=dialog_turn_inputs.tolis()[0],
                                                    action=generated_sequence[i].tolist()[0],
                                                    action_log_probs=generated_log_probs[i],
                                                    counselor=True,
                                                    human_response=False)
                past = outputs[1]
                outputs = model_A(expand_inputs_for_N_candidates(dialog_turn_inputs, num_candidates), past_, return_dict=False)
                past_ = outputs[1]
            else:
                #NOTE: Context will always be persuadee's utterance since, because candidates are generated in response to this utterance.
                outputs = model_A(dialog_turn_inputs, past, return_dict=False)
                past = outputs[1]
                outputs = model_A(expand_inputs_for_N_candidates(dialog_turn_inputs, num_candidates), past_, return_dict=False)
                past_ = outputs[1]
        
        context = dialog_turn_inputs.tolist()[0]
        cntxt = dialog_turn_inputs

    return dial_inputs, role_ids, scores_dict, #candidate_dict

def get_past(batches, model, device):
    
    states = torch.cat(batches, dim=-1).to(device)
    outputs = model(states, past_key_values=None, return_dict=False)
    
    return outputs[1]

def prepare_inputs_for_model(batches, model, num_candidates, device):
    
    states = get_history_utterances(batches, num_candidates)
    states = torch.cat(states, dim=1, device=device)
    outputs = model(states, past_key_values=None, return_dict=False)
    
    return outputs[1]

def get_history_utterances(batches, num_candidates):
    states = []
    for i in range(0, len(batches), num_candidates+1):
        states.append(i)
    return states

def get_recursive_past(dial_inputs, role_ids, model_A, model_B, device):
    '''
    Uses both models alternatively to calculate pasts.
    Used in case of training only the counselor.
    '''
    past = None
    for num_turn, utter in enumerate(dial_inputs):
        if role_ids[num_turn] == 0:
            _, past = model_A(utter.to(device), past_key_values=past, return_dict=False)
        else:
            _, past = model_B(utter.to(device), past_key_values=past, return_dict=False)
    return past
