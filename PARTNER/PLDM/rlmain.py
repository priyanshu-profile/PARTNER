import numpy as np
import torch.nn as nn
from nltk.translate.meteor_score import meteor_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import functools
import operator
import os
import pdb
import spacy
import pandas as pd
import json
import tqdm
import datetime
from tqdm.notebook import tqdm_notebook
import random
import pdb
from rlutils import collect_samples, ppo_step, generate_n_candidates, convert_sentences_to_strings, expand_inputs_for_N_candidates
from torch.utils.data import DataLoader, Dataset
from loss import SequenceCrossEntropyLoss
from ppo import PPOMemory
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup, RobertaForSequenceClassification, RobertaTokenizer   
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
from dataset import CounselingDataset

class Trainer():
    def __init__(self,
                 modelname,
                 csvfile,
                 n_epochs,
                 print_every,
                 learning_rate,
                 epsilon,
                 human_reward,
                 average_sent_loss,
                 device,
                 beta2,
                 num_candidates,
                 max_candidate_length,
                 top_p,
                 warmup_steps,
                 pad_token_id,
                 evaluate_every,
                 use_jaccard,
                 use_context,
                 act_num_labels,
                 mini_batch,
                 temperature,
                 use_recent_past,
                 recompute_log_prob,
                 use_counseling_classifier,
                 con_classifier_filename,
                 bin_classifier_filename,
                 gamma1,
                 gamma2,
                 gamma3,
                 gamma4,
                 gamma5,
                 train_single_model=False,
                 single_model_to_train=None,
                 loadModel=False,
                 batch_size=None,
                 loadFilename=None,
                 use_politeness_classifier=None,
                 pol_classifier_filename=None,
                 use_empathy_classifier=None,
                 emp_classifier_filename=None,
                 gammma5=None,
                 pol_num_labels=None,
                 emp_num_labels=None,
                 beta3=None,
                 beta4=None,
                 seedvalue=10):

        self.seedvalue = seedvalue
        self.train_single_model = train_single_model
        self.single_model_to_train = single_model_to_train
        self.nlp = spacy.load("en_core_web_sm")
        self.human_reward = human_reward
        self.seed(seedvalue)
        self.use_recent_past = use_recent_past
        self.temperature=temperature
        self.use_jacc = use_jaccard
        self.use_context = use_context

        self.average_sent_loss = average_sent_loss
        self.mini_batch = mini_batch
        self.evaluate_every = evaluate_every
        self.csvfile = csvfile
        self.modelname = modelname
        self.n_epochs = n_epochs
        self.print_every = print_every
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
        self.recompute_log_prob = recompute_log_prob
        self.num_candidates = num_candidates
        self.pad_token_id = pad_token_id
        self.max_candidate_length = max_candidate_length
        
        self.beta2  = beta2
        self.beta3 = beta3
        self.beta4 = beta4
        
        self.top_p = top_p
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.device = device

        self.num_labels = act_num_labels
        self.num_pol_labels = pol_num_labels
        self.num_emp_labels = emp_num_labels
        
        self.loadModel = loadModel
        self.loadFilename = loadFilename
        self.make_model_save_dir()
        self.make_stats_dir()
        
        self.use_counseling_classifier = use_counseling_classifier
        if self.use_counseling_classifier and con_classifier_filename:
            model_dict = torch.load(con_classifier_filename)
            self.counseling_classifier = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=self.num_labels)
            self.counseling_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
            self.counseling_classifier.config.problem_type = 'single_label_classification'
            self.counseling_classifier.load_state_dict(model_dict['state_dict'])
            self.counseling_classifier = self.counseling_classifier.to(self.device)
            self.counseling_classifier.eval()
            print('counseling Classifier Loaded! (in Evaluation Mode)')
            self.binary_classifier = None
        elif self.use_counseling_classifier and not con_classifier_filename:
            raise ValueError('counseling classifier use set to True, but filename to load from not defined.')
        else:
            self.counseling_classifier = None
            self.counseling_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

        self.use_politeness_classifier = use_politeness_classifier
        if self.use_politeness_classifier and pol_classifier_filename:
            model_dict = torch.load(pol_classifier_filename)
            self.politeness_classifier = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=self.num_pol_labels)
            self.politeness_classifier.config.problem_type = 'single_label_classification'
            self.politeness_classifier.load_state_dict(model_dict['state_dict'])
            self.politeness_classifier = self.politeness_classifier.to(self.device)
            self.politeness_classifier.eval()
        elif self.use_politeness_classifier and not pol_classifier_filename:
            raise ValueError('Politeness classifier use set to True, but filename to load from not defined.')
        else:
            self.politeness_classifier = None

        self.use_empathy_classifier = use_empathy_classifier
        if self.use_empathy_classifier and emp_classifier_filename:
            model_dict = torch.load(emp_classifier_filename)
            self.empathy_classifier = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=self.num_emp_labels)
            self.empathy_classifier.config.problem_type = 'single_label_classification'
            self.empathy_classifier.load_state_dict(model_dict['state_dict'])
            self.empathy_classifier = self.empathy_classifier.to(self.device)
            self.empathy_classifier.eval()
        elif self.use_empathy_classifier and not emp_classifier_filename:
            raise ValueError('Empathy classifier use set to True, but filename to load from not defined.')
        else:
            self.empathy_classifier = None

        self.getDataset()
        
        self.initialize_models()
        self.configure_optimizer()
        
        self.buffer_memory = PPOMemory()
        
        self.saveModelConfig()
        self.criterion = SequenceCrossEntropyLoss()

        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.gamma4 = gamma4
        self.gamma5 = gamma5

        self.x_num = {}
        self.count_dict_str = {}
        self.count_dict_num = {}
        self.act_dict_num_to_str = {}
        
        self.con_classifier_filename = con_classifier_filename
        self.pol_classifier_filename = pol_classifier_filename
        self.pol_classifier_filename = emp_classifier_filename

        self.act_dict = label_dict = {
                                            "Counselling support": 0,
                                            "Legal assistance": 1,
                                            "Persuasion": 2,
                                            "Seek information": 3,
                                            "Deliver information": 4,
                                            "Re-check assistance": 5,
                                            "Greet": 6,
                                            "Closing remark": 7
                                        }
        self.initialize_act_count()

    def initialize_classifier_models(self):
        model_dict = torch.load(self.con_classifier_filename)
        self.counseling_classifier = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=self.num_labels)
        self.counseling_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.counseling_classifier.config.problem_type = 'single_label_classification'
        self.counseling_classifier.load_state_dict(model_dict['state_dict'])
        self.counseling_classifier = self.counseling_classifier.to(self.device)
        self.counseling_classifier.eval()
        print('counseling Classifier Loaded! (in Evaluation Mode)')

    def saveModelConfig(self):
        if self.train_single_model:
            config_model_train = self.single_model_to_train
            print('Training Only :', self.single_model_to_train)
        else:
            config_model_train = 'Both Models being Trained.'
            print('Both Models being Trained.')
        config = {'Basic Info': [datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S")],
                  'NOTES': 'GPT2-MEDIUM',
                  'modelname': self.modelname,
                  'Training only one Model': self.train_single_model,
                  'Training Models': config_model_train,
                  'counseling_classifier': self.use_counseling_classifier,
                  'politeness_classifier': self.use_politeness_classifier,
                  'empathy_classifier': self.use_empathy_classifier,
                  'num_labels_counseling': self.num_labels,
                  'num_labels_politeness': self.num_pol_labels,
                  'num_labels_empathy': self.num_emp_labels,
                  'beta2': self.beta2,
                  'beta3': self.beta3,
                  'beta4': self.beta4,
                  'device': self.device,
                  'use_jaccard_similarity': self.use_jacc,
                  'use_context' : self.use_context,
                  'modelLoaded': self.loadFilename,
                  'human_reward': self.human_reward,
                  'average_sent_loss' : self.average_sent_loss,
                  'n_epochs': self.n_epochs,
                  'use_recent_past': self.use_recent_past,
                  'temperature': self.temperature,
                  'learning_rate': self.learning_rate,
                  'epsilon': self.epsilon,
                  'num_candidates': self.num_candidates,
                  'pad_token_id': self.pad_token_id,
                  'max_candidate_length': self.max_candidate_length,
                  'recompute_log_prob': self.recompute_log_prob,
                  'evaluate_every': self.evaluate_every,
                  'top_p': self.top_p,
                  'warmup_steps': self.warmup_steps,
                  'batch_size':self.batch_size,
                  'seed': self.seedvalue}
        configfilename = os.path.join(self.savefolder, self.modelname, 'config')
        if not os.path.exists(configfilename):
            os.makedirs(configfilename)
        configfilename = configfilename + '/config' + '_' + self.modelname + '.json'
        with open(configfilename, 'w') as f:
            json.dump(config, f)

    def seed(self,seed=10):
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
          
    def convertDicttoList(self, data: dict):
        return list(data.values())

    def random_split_data(self, data):
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        train_data = [data[idx] for idx in indices[:800]]
		val_data = [data[idx] for idx in indices[800:]]
        
        return train_data, val_data

    def getDataset(self):
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        data = self.extract_data(self.csvfile)
        data = self.utteranceToConversation(self.csvfile, data)
        
        self.traindata, self.valdata = self.random_split_data(data)
        
        if self.counseling_classifier and self.politeness_classifier and self.empathy_classifier:
            use_counseling_labels=True
            use_politeness_labels=True
            use_empathy_labels=True
        elif  not self.counseling_classifier and not self.politeness_classifier and self.empathy_classifier:
        	use_counseling_labels=False
        	use_politeness_labels=False
        	use_empathy_labels=True
        elif  not self.counseling_classifier and self.politeness_classifier and not self.empathy_classifier:
        	use_counseling_labels=False
        	use_politeness_labels=True
        	use_empathy_labels=False
        elif  not self.counseling_classifier and self.politeness_classifier and self.empathy_classifier:
        	use_counseling_labels=False
        	use_politeness_labels=True
        	use_empathy_labels=True
        elif self.counseling_classifier and not self.politeness_classifier and not self.empathy_classifier:
        	use_counseling_labels=True
        	use_politeness_labels=False
        	use_empathy_labels=False
        elif self.counseling_classifier and not self.politeness_classifier and self.empathy_classifier:
        	use_counseling_labels=True
        	use_politeness_labels=False
        	use_empathy_labels=True
        elif self.counseling_classifier and self.politeness_classifier and not self.empathy_classifier:
        	use_counseling_labels=True
        	use_politeness_labels=True
        	use_empathy_labels=False
        else:
        	use_counseling_labels=False
        	use_politeness_labels=False
        	use_empathy_labels=False
        
        traindata_ = CounselingDataset(self.traindata,
                                     self.tokenizer,
                                     use_counseling_labels=use_counseling_labels,
                                     use_politeness_labels=use_politeness_labels,
                                     use_empathy_labels=use_empathy_labels)
        
        self.turn_ending = traindata_.get_turn_ending()
        
        valdata_ = CounselingDataset(self.valdata,
                                   self.tokenizer,
                                   use_counseling_labels=use_counseling_labels,
                                   use_politeness_labels=use_politeness_labels,
                                   use_empathy_labels=use_empathy_labels)
        
        self.train_dataloader = DataLoader(dataset=traindata_,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           collate_fn=traindata_.collate)
        
        self.val_dataloader = DataLoader(dataset=valdata_,
                                         shuffle=False,
                                         batch_size=self.batch_size,
                                         collate_fn=valdata_.collate)

    def initialize_models(self):
        if not self.train_single_model:
            self.model_A = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model_B = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model_A_ref = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model_B_ref = GPT2LMHeadModel.from_pretrained("gpt2")
        else:
            if self.single_model_to_train == 'counselor':
                self.model_A = GPT2LMHeadModel.from_pretrained("gpt2")
                self.model_A_ref = GPT2LMHeadModel.from_pretrained("gpt2")
            else:
                self._model_B = GPT2LMHeadModel.from_pretrained("gpt2")
                self.model_B_ref = GPT2LMHeadModel.from_pretrained("gpt2")

        if self.loadModel:
            if self.loadFilename:
                model_A_state_dict, model_B_state_dict = torch.load(self.loadFilename)#, map_location=self.device)
                if not self.train_single_model:
                    self.model_A.load_state_dict(model_A_state_dict)
                    self.model_A_ref.load_state_dict(model_A_state_dict)
                    self.model_B.load_state_dict(model_B_state_dict)
                    self.model_B_ref.load_state_dict(model_B_state_dict)
                    self.model_A = self.model_A.to(self.device)
                    self.model_A_ref = self.model_A_ref.to(self.device)
                    self.model_B = self.model_B.to(self.device)
                    self.model_B_ref = self.model_B_ref.to(self.device)
                    self.model_A.train()
                    self.model_B.train()
                    self.model_A_ref.eval()
                    self.model_B_ref.eval()
                else:
                    if self.single_model_to_train == 'counselor':
                        self.model_A.load_state_dict(model_A_state_dict)
                        self.model_A_ref.load_state_dict(model_A_state_dict)
                        self.model_A = self.model_A.to(self.device)
                        self.model_A_ref = self.model_A_ref.to(self.device)
                        self.model_A.train()
                        self.model_A_ref.eval()
                        self.model_B = None
                        self.model_B_ref = None
                    else:
                        self.model_B.load_state_dict(model_B_state_dict)
                        self.model_B_ref.load_state_dict(model_B_state_dict)
                        self.model_B = self.model_B.to(self.device)
                        self.model_B_ref = self.model_B_ref.to(self.device)
                        self.model_B.train()
                        self.model_B_ref.eval()
                        self.model_A = None
                        self.model_A_ref = None
                print('\n')
                print("Models loaded from file ", self.loadFilename)
            else:
                print('Models not loaded since directory not provided.')
        print(f"Models Initalized!")
        print('\n')


    def configure_optimizer(self):
        
        self.num_train_optimization_steps = self.n_epochs * len(self.traindata) # // self.batch_size

        if not self.train_single_model:
            param_optimizer = list(self.model_A.named_parameters()) + list(self.model_B.named_parameters())
        else:
            if self.single_model_to_train == 'counselor':
                param_optimizer = list(self.model_A.named_parameters())
        no_decay = ['bias', 'ln', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = optimizer = AdamW(optimizer_grouped_parameters,
                                           lr=self.learning_rate,
                                           eps=1e-06)

    def initialize_act_count(self):

        self.x_num = {}
        self.count_dict_str = {}
        self.act_dict_num_to_str = {}
        for i in self.act_dict:
            self.count_dict_num[self.act_dict[i]] = 0
            self.count_dict_str[i] = 0
            self.act_dict_num_to_str[self.act_dict[i]] = i


    def get_candidate_lengths(self, candidate_dict):

        avg_iter_length = []
        for i in candidate_dict:
            for j in candidate_dict[i]:
                 avg_iter_length.append(len(j.split()))
        return avg_iter_length

    def get_num_candidate_with_act(self, candidate_dict):

        pred_labels = []
        for ref in candidate_dict:
            inputs = self.counseling_tokenizer(candidate_dict[ref], padding=True, truncation=True, return_tensors='pt')
            output = self.politeness_classifier(inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device))
            probs = F.softmax(output.logits, dim=1)
            _, pred_label = torch.topk(probs, k=1, dim=-1)
            pred_labels.extend(pred_label.squeeze(1).tolist())
        num_act = np.mean(np.array(pred_labels) != 0)
        for i in pred_labels:
            self.count_dict_num[i] += 1
        for j in self.count_dict_num:
            self.count_dict_str[self.act_dict_num_to_str[j]] = self.count_dict_num[j]
        return num_act

    def get_num_candidate_with_empathy(self, candidate_dict):

        predicted_label = []
        
        for cand in candidate_dict:
            pred_label = self.model2.predict(candidate_dict[cand])[0]
            predicted_label.extend(pred_label)
        
        num_emp_str = np.mean(predicted_label)
        
        return num_emp_str

    def get_pol_pr(self, candidate_dict, gt_pol_str):

        pred_labels = []
        true_labels = []
        
        for idx, ref in enumerate(candidate_dict):
            inputs = self.counseling_tokenizer(candidate_dict[ref], padding=True, truncation=True, return_tensors='pt')
            
            output = self.politeness_classifier(inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device))
            
            probs = F.softmax(output.logits, dim=1)
            
            _, pred_label = torch.topk(probs, k=1, dim=-1)
            
            pred_labels.extend(pred_label.squeeze(1).tolist())

            for i in range(len(candidate_dict[ref])): true_labels.append(gt_pol_str[idx])
        
        num_pol = 0
        for gt, pt in zip(true_labels, pred_labels):
            if gt == pt:
                num_pol += 1
        pdb.set_trace() 
        return num_pol / len(pred_labels)


    def validate_model(self, dataloader):

        with torch.no_grad():
            if not self.train_single_model:
                self.model_A.eval()
                self.model_B.eval()
            else:
                if self.single_model_to_train == 'counselor':
                    self.model_A.eval()
                else:
                    self.model_B.eval()

            with torch.no_grad():
                
                progress_bar = tqdm_notebook
                pbar = progress_bar(dataloader)
               
                total_ppl = []
                total_loss = []
                candidates_dict = {}

                for batch in pbar:

                    if sum([len(item) for item in batch[0][1]]) > 1024:
                        continue

                    if not self.counseling_classifier and not self.politeness_classifier and self.empathy_classifier:
                        role_ids, dialog_tokens, empathy_label = batch[0]
                    elif not self.counseling_classifier and self.politeness_classifier and not self.empathy_classifier:
                        role_ids, dialog_tokens, politeness_label = batch[0]
                    elif not self.counseling_classifier and self.politeness_classifier and self.empathy_classifier:
                        role_ids, dialog_tokens, politeness_label, empathy_label  = batch[0]
                    elif self.counseling_classifier and not self.politeness_classifier and not self.empathy_classifier:
                        role_ids, dialog_tokens, counseling_label  = batch[0]
                    elif self.counseling_classifier and not self.politeness_classifier and self.empathy_classifier:
                        role_ids, dialog_tokens, counseling_classifier, counseling_label, empathy_label  = batch[0]
                    elif self.counseling_classifier and self.politeness_classifier and not self.empathy_classifier:
                        role_ids, dialog_tokens, counseling_classifier, counseling_label, politeness_label = batch[0]
                    elif self.counseling_classifier and self.politeness_classifier and self.empathy_classifier:
                        role_ids, dialog_tokens, counseling_classifier, counseling_label, politeness_label, empathy_label  = batch[0]
                    else:
                    	role_ids, dialog_tokens = batch[0]

                    dial_inputs = [torch.LongTensor(item).unsqueeze(0) for item in dialog_tokens]
                    past = None
                    past_ = None
                    all_logits = []
                    target = []

                    for turn_num, dial_turn_inputs in enumerate(dial_inputs):

                        if not self.train_single_model:
                            if role_ids[turn_num] == 0:
                                outputs = self.model_A(dial_turn_inputs, past_key_values=past, return_dict=False)
                                past = outputs[1]
                                all_logits.append(outputs[0])
                            else:
                                outputs = self.model_B(dial_turn_inputs, past_key_values=past, return_dict=False)
                                past = outputs[1]
                                all_logits.append(outputs[0])
                        else:
                            if self.single_model_to_train == 'counselor':
                                if role_ids[turn_num] == 0:
                                    dial_turn_inputs = dial_turn_inputs.to(self.device)
                                    outputs = self.model_A(dial_turn_inputs, past_key_values=past, return_dict=False)
                                    past = outputs[1]
                                    all_logits.append(outputs[0])
                                    target.append(dial_turn_inputs)
                                    
                    all_logits = torch.cat(all_logits, dim=1)
                    all_logits = all_logits[:, :-1].contiguous()

                    if not self.train_single_model:
                        target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
                    else:
                        target = torch.cat(target, dim=1)[:, 1:].contiguous()
                    
                    target_mask = torch.ones_like(target).float()

                    loss = self.criterion(all_logits, target, target_mask, label_smoothing=-1, reduce='sentence')
                    total_loss.extend(loss.tolist())

                    ppl = torch.exp(loss)
                    total_ppl.extend(ppl.tolist())
                    
                print('\n')
                print(f"Validation Perplexity: {np.mean(total_ppl)}")

        return np.mean(total_ppl), np.mean(total_loss)
    

    def make_stats_dir(self):
        
        self.statsfolder = os.path.join(os.getcwd(), self.savefolder, self.modelname, 'stats')
        if not os.path.exists(self.statsfolder):
            os.makedirs(self.statsfolder)


    def make_model_save_dir(self):
        self.savefolder = os.path.join(os.getcwd(), 'models/RL')
        if not os.path.exists(self.savefolder):
            print("Model save folder doesn't exist.")
            os.makedirs(self.savefolder)
            print(f"Created folder {self.savefolder} to save the models.")


    def save_models(self, num_iter):
        
        modeldir = os.path.join(self.savefolder, self.modelname)
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
            print('Created Directory for saving models!')
        filename = modeldir + '/' + self.modelname + '_' + str(num_iter) + ".pth"
        torch.save(self.model_A.state_dict(), filename)

    def modified_train_one_iter(self, batch):
        dial_inputs, role_ids, scores_dict = collect_samples(batch,
                                                             model_A=self.model_A_ref,
                                                             model_B=self.model_B,
                                                             top_p=self.top_p,
                                                             eos_token_id=self.turn_ending[0],
                                                             pad_token_id=self.turn_ending[1],
                                                             average_sent_loss=self.average_sent_loss,
                                                             max_gen_length=self.max_candidate_length,
                                                             buffer_memory=self.buffer_memory,
                                                             use_context=self.use_context,
                                                             device=self.device,
                                                             num_candidates=self.num_candidates,
                                                             human_reward=self.human_reward,
                                                             use_jacc=self.use_jacc,
                                                             counseling_tokenizer=self.counseling_tokenizer,
                                                             counseling_classifier=self.counseling_classifier,
                                                             politeness_classifier=self.politeness_classifier,
                                                             empathy_classifier=self.empathy_classifier,
                                                             beta2=self.beta2,
                                                             beta3=self.beta3,
                                                             beta4=self.beta4,
                                                             tokenizer=self.tokenizer,
                                                             criterion=self.criterion,
                                                             temperature=self.temperature,
                                                             use_recent_past=self.use_recent_past,
                                                             recompute_log_prob=self.recompute_log_prob,
                                                             nlp=self.nlp,
                                                             train_single_model=self.train_single_model,
                                                             model_to_train=self.single_model_to_train,
                                                             gamma1=self.gamma1,
                                                             gamma2=self.gamma2,
                                                             gamma3=self.gamma3,
                                                             gamma4=self.gamma4,
                                                             gamma5=self.gamma5)

        log_dict = ppo_step(model_A=self.model_A,
                            model_B=self.model_B,
                            buffer_memory=self.buffer_memory,
                            train_single_model=self.train_single_model,
                            dial_inputs= dial_inputs,
                            model_to_train=self.single_model_to_train,
                            device=self.device,
                            ppo_epsilon=self.epsilon,
                            num_candidates=self.num_candidates,
                            use_recent_past=self.use_recent_past,
                            average_sent_loss=self.average_sent_loss,
                            criterion=self.criterion,
                            optimizer=self.optimizer,
                            role_ids=role_ids)

        self.buffer_memory.clear_memory()

        return log_dict, scores_dict 
 
    def train(self):

        update_count = 0
        progress_bar = tqdm_notebook

        val_ppl = []
        val_loss = []

        rewards = []
        kl = []
        clip_frac = []

        cos_jacc_scores = []
        counseling_scores = []
        politeness_scores = []
        empathy_scores = []
        context_coherence_scores = []
        
        con_actual_probs = []
        con_other_probs = []

        pol_actual_probs = []
        pol_other_probs = []

        emp_actual_probs = []
        emp_other_probs = []
        

        best_ppl = None
        

        
        iters = None
        

        pbar = progress_bar(self.train_dataloader)

        for i in range(self.n_epochs):
            if not self.train_single_model:
                self.model_A.train()
                self.model_B.train()
            else:
                if self.single_model_to_train == 'counselor':
                    self.model_A.train()
            for batch in pbar:
                if sum([len(item) for item in batch[0][1]]) > 1024 - self.max_candidate_length:
                    continue

                print(f"ITERATION: {update_count}")

                batch = batch[0]
                log_dict, scores_dict  = self.modified_train_one_iter(batch)

                clip_frac.append(log_dict['clip_frac'])
                kl.append(log_dict['approx_kl'])
                rewards.append(log_dict['reward'])

                cos_jacc_scores.extend(scores_dict['cos_jacc_scores'])
                counseling_scores.extend(scores_dict['counseling_scores'])
                politeness_scores.extend(scores_dict['politeness_scores'])
                empathy_scores.extend(scores_dict['empathy_scores'])
                context_coherence_scores.extend(scores_dict['context_coherence_scores'])
                
                con_actual_probs.extend(scores_dict['counseling_actual_prob'])
                con_other_probs.extend(scores_dict['counseling_other_prob'])

                pol_actual_probs.extend(scores_dict['politeness_actual_prob']) 
                pol_other_probs.extend(scores_dict['politeness_other_prob'])

                emp_actual_probs.extend(scores_dict['empathy_actual_prob']) 
                emp_other_probs.extend(scores_dict['empathy_other_prob'])

                np.save(self.statsfolder + '/' + 'contextual_coherence_scores.npy', np.array(context_coherence_scores))
                np.save(self.statsfolder + '/' + 'non_rep_scores.npy', np.array(cos_jacc_scores))

                if self.counseling_classifier:
                    np.save(self.statsfolder + '/' + 'counseling_scores.npy', np.array(counseling_scores))
                    np.save(self.statsfolder + '/' + 'counseling_actual_prob.npy', np.array(con_actual_probs))
                    np.save(self.statsfolder + '/' + 'counseling_other_prob.npy', np.array(con_other_probs))
                if self.politeness_classifier:
                    np.save(self.statsfolder + '/' + 'politeness_scores.npy', np.array(politeness_scores))
                    np.save(self.statsfolder + '/' + 'politeness_actual_prob.npy', np.array(pol_actual_probs))
                    np.save(self.statsfolder + '/' + 'politeness_other_prob.npy', np.array(pol_other_probs))
                if self.empathy_classifier:
                    np.save(self.statsfolder + '/' + 'empathy_scores.npy', np.array(empathy_scores))
                    np.save(self.statsfolder + '/' + 'empathy_actual_prob.npy', np.array(emp_actual_probs))
                    np.save(self.statsfolder + '/' + 'empathy_other_prob.npy', np.array(emp_other_probs))                    
                
                update_count += 1

                if  update_count % self.evaluate_every == 0:
                    
                    ppl, loss = self.validate_model(self.val_dataloader)
                    
                    if best_ppl is None:

                        best_ppl = ppl
                        iters = update_count

                        
                        if update_count > 20 and update_count < 22:
                          self.save_models(iters)
                          print(f'Saving Model at {iters}')
                        
                        
                    else:
                        if ppl < best_ppl:
                            best_ppl = ppl
                            iters = update_count

                        if update_count > 20 and update_count < 22:
                          self.save_models(iters)
                          print(f'Saving Model at {iters}')
                
                    print('\n')
                    print(f'Best Perplexity Found so far {best_ppl} for iteration: {iters}')
                    print('\n')
                    
                    val_ppl.append(ppl)
                    val_loss.append(loss)
                    
                                
                    np.save(self.statsfolder + '/' + 'val_PPL_iter'  + '.npy', np.array(val_ppl))
                    
                    
                    np.save(self.statsfolder + '/' + 'train_rewards' + '.npy', np.array(rewards))
                    np.save(self.statsfolder + '/' + 'train_kl' + '.npy', np.array(kl))
                    np.save(self.statsfolder + '/' + 'train_clip_frac' + '.npy', np.array(clip_frac))
                    np.save(self.statsfolder + '/' + 'best_ppl_iteration_value' + '.npy', np.array(iters))
                    
                    
                    if not self.train_single_model:
                        self.model_A.train()
                        self.model_B.train()
                    else:
                        if self.single_model_to_train == 'counselor':
                            self.model_A.train()
                
        return best_ppl, iters

if __name__ == '__main__':
    trainer = Trainer(modelname='DESIRED MODEL NAME TO SAVE RL-FINE-TUNED MODEL',
                      csvfile='PATH TO DATASET',
                      device='cuda',
                      n_epochs=20,
                      batch_size=1,
                      mini_batch=20,
                      train_single_model=True,
                      single_model_to_train= 'counselor',
                      num_candidates=3,
                      recompute_log_prob=True,
                      average_sent_loss=True,
                      max_candidate_length=50,
                      human_reward=10,
                      beta2=2.0,
                      beta3=2.0,
                      beta4=2.0,
                      top_p=0.9,
                      temperature=0.8,
                      use_recent_past=True,
                      warmup_steps=10,
                      print_every=1,
                      evaluate_every=1,
                      learning_rate=2e-5,
                      epsilon=0.2,
                      loadModel=True,
                      loadFilename="PATH TO MLE LOSS BASED TRAINED DIALOGUE MODEL",
                      pad_token_id=2,
                      seedvalue=10, # 10 should be the seed value since pre trained on the same seed. 
                      use_counseling_classifier=True,
                      use_politeness_classifier=True,
                      use_empathy_classifier=True,
                      con_classifier_filename= "PATH TO COUNSELING CLASSIFIER",
                      bin_classifier_filename="PATH TO BINARY COUNSELING CLASSIFIER",
                      pol_classifier_filename="PATH TO POLITENESS CLASSIFIER",
                      emp_classifier_filename="PATH TO EMPATHY CLASSIFIER",
                      act_num_labels=8,
                      pol_num_labels=3,
                      emp_num_labels=7,
                      use_jaccard=True,
                      use_context=True,
                      gamma1=0.3,
                      gamma2=0.2,
                      gamma3=0.2,
                      gamma4=0.2,
                      gamma5=0.1)
    trainer.train()
