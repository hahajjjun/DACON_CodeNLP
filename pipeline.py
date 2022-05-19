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

warnings.filterwarnings("ignore")

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

#------------ Assembly -------------#
tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
model = AutoModel.from_pretrained("huggingface/CodeBERTa-small-v1")
train, valid, test, ss = load_data('/home/hahajjjun/Junha Park/DACON_PYNLI')
train = text_clean(train)
valid = text_clean(valid)
device = torch.device("cuda:0")
lr = 2e-5
batch_size= 32
warmup_ratio = 0.06
num_epochs = 2
log_interval = 10000

model = codeBERTclassifier(model).to(device)
optimizer = AdamW(model.parameters(), lr=lr)
best_models = []
train_dataset = customDataset(train,'train')
valid_dataset = customDataset(valid,'train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

total_steps = len(train_loader) * num_epochs
warmup_step = int(total_steps * warmup_ratio)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
valid_loss_min = 0.4
valid_acc_max = 0.8
    
for epoch in range(num_epochs):
    batches = 0
    total_loss = 0.0
    correct = 0
    total =0
    model.train()

    for input_ids_batch, attention_masks_batch, y_batch in tqdm(train_loader): # token_type_batch, y_batch in tqdm(train_loader):
        optimizer.zero_grad()
        y_batch = y_batch.to(device)
        y_pred = model(input_ids_batch.to(device), attention_masks_batch.to(device))
        loss = F.cross_entropy(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(y_pred, 1)
        correct += (predicted == y_batch).sum()
        total += len(y_batch)
        batches += 1
        if batches % log_interval == 0:
            print("Batch Loss: ", total_loss / batches, "Accuracy: ", correct.float() / total)
    scheduler.step()
    val_loss = []
    val_acc = []
    
    # VALIDATION #
    for input_ids_batch, attention_masks_batch, y_batch in tqdm(valid_loader):

        model.eval()
        with torch.no_grad():
            y_batch = y_batch.to(device)
            y_pred = model(input_ids_batch.to(device), attention_masks_batch.to(device))
            valid_loss = F.cross_entropy(y_pred, y_batch)
            valid_loss = valid_loss.cpu().detach().numpy()
            preds = torch.argmax(y_pred,1)
            preds = preds.cpu().detach().numpy()
            y_batch = y_batch.cpu().detach().numpy()
            batch_acc = (preds==y_batch).mean()
            val_loss.append(valid_loss)
            val_acc.append(batch_acc)


    val_loss = np.mean(val_loss)
    val_acc = np.mean(val_acc)

    print(f'Epoch: {epoch} - valid Loss: {val_loss:.6f} - valid_acc : {val_acc:.6f}')
    if valid_acc_max < val_acc:
        valid_acc_max = val_acc
        best_models.append(model)
        torch.save(model.state_dict(), f'/home/hahajjjun/Junha Park/DACON_PYNLI/weights/codeBERT.pth') 
        print('model saved, model val acc : ',val_acc)
        print('best models size : ',len(best_models))