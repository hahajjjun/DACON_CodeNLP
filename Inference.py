from transformers import AutoTokenizer, AutoModel
import torch
import random
import numpy as np
import os
import pandas as pd
import torch.nn as nn
import warnings 
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
seed_everything(42)

def load_data(path):
    #TRAIN = os.path.join(path, 'train.csv')
    TRAIN = os.path.join(path, 'train_data.csv')
    VALID = os.path.join(path, 'val_data.csv')
    TEST = os.path.join(path, 'test.csv')
    SS = os.path.join(path, 'sample_submission.csv')
    train = pd.read_csv(TRAIN)
    valid = pd.read_csv(VALID)
    test = pd.read_csv(TEST)
    sample_submission = pd.read_csv(SS)
    return train,valid,test,sample_submission
    #return train, test, sample_submission
    
def text_clean_test(df):
    df["text_sum"] = "[CLS] "+df["code1"]+" [SEP] "+df["code2"]+" [SEP]"
    df = df[['text_sum', 'code1']]
    return df

def reverse_text_clean_test(df):
    df["text_sum"] = "[CLS] "+df["code2"]+" [SEP] "+df["code1"]+" [SEP]"
    df = df[['text_sum','code1']]
    return df

def text_clean(df):
    df["text_sum"] = "[CLS] "+df["code1"]+" [SEP] "+df["code2"]+" [SEP]"
    df = df[['text_sum','similar']]
    return df

def codebert_transform(text):
    transform = tokenizer(text,pad_to_max_length=True,truncation=True,max_length=512, return_tensors='pt',add_special_tokens=False)
    return transform

#------ Dataset -------#
class customDataset(Dataset):
    def __init__(self,dataset,mode='train',transform=codebert_transform):
        super(customDataset, self).__init__()
        self.mode = mode
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, idx):
        text = self.dataset['text_sum'].iloc[idx]
        tokens = self.transform(text)
        token_ids = tokens['input_ids'][0]
        attn_masks = tokens['attention_mask'][0]
        #token_type_ids = tokens['token_type_ids'][0]

        if self.mode == 'test':
          return token_ids,attn_masks #,token_type_ids
        else: 
          labels = self.dataset['similar'].iloc[idx]
          return token_ids,attn_masks,labels #token_type_ids, labels

    def __len__(self):
        return(len(self.dataset))

#------- Model ------#
class codeBERTclassifier(nn.Module):
    def __init__(self, model, hidden_size = 768, num_classes=2, params=None,  freeze=False):
        super(codeBERTclassifier, self).__init__()
        self.model = model
        self.freeze = freeze

        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False


        self.classifier = nn.Linear(hidden_size , 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_layer1 = nn.Linear(256,128)
        self.fc_layer2 = nn.Linear(128,num_classes)
        self.softmax = nn.Softmax()


    def forward(self, input_ids, attn_masks):

        _,pooler = self.model(input_ids, attn_masks, return_dict=False)
        output1 = self.classifier(pooler)
        output2 = self.fc_layer1(output1)
        output3 = self.fc_layer2(self.dropout(output2))
        return output3
    
tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
model = AutoModel.from_pretrained("huggingface/CodeBERTa-small-v1")
train, valid, test, ss = load_data('/home/hahajjjun/Junha Park/DACON_PYNLI')
test_forward = text_clean_test(test)
test_backward = reverse_text_clean_test(test)
test_forward_dataset = customDataset(test_forward, 'test')
test_backward_dataset = customDataset(test_backward, 'test')

device = torch.device("cuda:0")
test_forward_dataloader = DataLoader(test_forward_dataset, batch_size=64, shuffle=False)
test_backward_dataloader = DataLoader(test_backward_dataset, batch_size=64, shuffle=False)
model = codeBERTclassifier(model).to(device)
model.load_state_dict(torch.load(f'/home/hahajjjun/Junha Park/DACON_PYNLI/weights/codeBERT.pth'))
model.eval()

answer_forward = []
with torch.no_grad():
    for input_ids_batch, attention_masks_batch in tqdm(test_forward_dataloader):
        y_pred = model(input_ids_batch.to(device), attention_masks_batch.to(device)).detach().cpu().numpy()
        answer_forward.extend(y_pred)
    
answer_backward = []
with torch.no_grad():
    for input_ids_batch, attention_masks_batch in tqdm(test_backward_dataloader):
        y_pred = model(input_ids_batch.to(device), attention_masks_batch.to(device)).detach().cpu().numpy()
        answer_backward.extend(y_pred)
        
p = []
for idx in range(len(answer_forward)):
    p.append((answer_forward[idx]+answer_backward[idx]).argmax())
ss['similar'] = p
ss.to_csv('submission_codeBERTa.csv', index = False)