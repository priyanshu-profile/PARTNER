import torch
from torch.utils.data import DataLoader, Dataset
import pdb

class CounselingDataset(Dataset):
    def __init__(self, data, tokenizer, use_politeness_labels=False, use_empathy_labels=False, use_counseling_labels=False):
        self.data = data
        self.use_politeness_labels = use_politeness_labels
        self.use_empathy_labels = use_empathy_labels
        self.use_counseling_labels = use_counseling_labels        
        self.tokenizer = tokenizer
        self.tokenizer.max_len = 1500
        self.turn_ending = [628, 198]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):         

        if self.use_counseling_labels and self.use_politeness_labels and not self.use_empathy_labels:
            dial_tokens = [self.tokenizer.encode(item[0]) + self.turn_ending for item in self.data[index]]
            role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
            con_labels = [item[1] for item in self.data[index]]
            politeness_labels = [item[2] for item in self.data[index]]
            return role_ids, dial_tokens, con_labels, politeness_labels
        elif self.use_counseling_labels and self.use_politeness_labels and self.use_empathy_labels:
            dial_tokens = [self.tokenizer.encode(item[0]) + self.turn_ending for item in self.data[index]]
            # 32 is the encoding for "A:"
            role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
            con_labels = [item[1] for item in self.data[index]]
            politeness_labels = [item[2] for item in self.data[index]]
            empathy_labels = [item[3] for item in self.data[index]]
            return role_ids, dial_tokens, con_labels, politeness_labels, empathy_labels
        elif self.use_counseling_labels and not self.use_politeness_labels and not self.use_empathy_labels:
            dial_tokens = [self.tokenizer.encode(item[0]) + self.turn_ending for item in self.data[index]]
            # 32 is the encoding for "A:"
            role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
            con_labels = [item[1] for item in self.data[index]]
            return role_ids, dial_tokens, con_labels
        elif not self.use_counseling_labels and self.use_politeness_labels and not self.use_empathy_labels:
            dial_tokens = [self.tokenizer.encode(item[0]) + self.turn_ending for item in self.data[index]]
            # 32 is the encoding for "A:"
            role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
            politeness_labels = [item[1] for item in self.data[index]]
            return role_ids, dial_tokens, politeness_labels
        elif self.use_counseling_labels and not self.use_politeness_labels and self.use_empathy_labels:
            dial_tokens = [self.tokenizer.encode(item[0]) + self.turn_ending for item in self.data[index]]
            role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
            con_labels = [item[1] for item in self.data[index]]
            empathy_labels = [item[2] for item in self.data[index]]
            return role_ids, dial_tokens, con_labels, empathy_labels
        elif not self.use_counseling_labels and self.use_politeness_labels and self.use_empathy_labels:
            dial_tokens = [self.tokenizer.encode(item[0]) + self.turn_ending for item in self.data[index]]
            role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
            politeness_labels = [item[1] for item in self.data[index]]
            empathy_labels = [item[2] for item in self.data[index]]
            return role_ids, dial_tokens, politeness_labels, empathy_labels
        elif not self.use_counseling_labels and not self.use_politeness_labels and self.use_empathy_labels:
            dial_tokens = [self.tokenizer.encode(item[0]) + self.turn_ending for item in self.data[index]]
            role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
            empathy_labels = [item[1] for item in self.data[index]]
            return role_ids, dial_tokens, empathy_labels
        else:
            dial_tokens = [self.tokenizer.encode(item) + self.turn_ending for item in self.data[index]]
            # 32 is the encoding for "A:"
            role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
            return role_ids, dial_tokens



        if self.use_politeness_labels and not self.use_empathy_labels: # per
            dial_tokens = []
            politeness_labels = [item[1] for item in self.data[index]]
            for i in range(len(self.data[index])):
              item1 = self.data[index][i][0]
              sep = "\t"
              item2 = self.persona[index][i]
              dial_tokens.append(self.tokenizer.encode(item1)+self.tokenizer.encode(sep)+self.tokenizer.encode(item2)+self.turn_ending)
            role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
            return role_ids, dial_tokens, politeness_labels
        elif self.use_politeness_labels and self.use_empathy_labels: # per and pol
            dial_tokens = []
            politeness_labels = [item[1] for item in self.data[index]]
            empathy_labels = [item[2] for item in self.data[index]]
            for i in range(len(self.data[index])):
              item1 = self.data[index][i][0]
              sep = "\t"
              item2 = self.persona[index][i]
              dial_tokens.append(self.tokenizer.encode(item1)+self.tokenizer.encode(sep)+self.tokenizer.encode(item2)+self.turn_ending)
            role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
            return role_ids, dial_tokens, politeness_labels, empathy_labels
        elif not self.use_politeness_labels and not self.use_empathy_labels: # no per no pol
            dial_tokens = []
            for i in range(len(self.data[index])):
              item1 = self.data[index][i][0]
              sep = "\t"
              item2 = self.persona[index][i]
              dial_tokens.append(self.tokenizer.encode(item1)+self.tokenizer.encode(sep)+self.tokenizer.encode(item2)+self.turn_ending)
            role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
            return role_ids, dial_tokens
        elif not self.use_politeness_labels and self.use_empathy_labels: # pol 
            dial_tokens = []
            empathy_labels = [item[1] for item in self.data[index]]
            for i in range(len(self.data[index])):
              item1 = self.data[index][i][0]
              sep = "\t"
              item2 = self.persona[index][i]
              dial_tokens.append(self.tokenizer.encode(item1)+self.tokenizer.encode(sep)+self.tokenizer.encode(item2)+self.turn_ending)
            role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
            return role_ids, dial_tokens, empathy_labels

    def collate(self, unpacked_data):
        return unpacked_data

    def get_turn_ending(self):
        return self.turn_ending
