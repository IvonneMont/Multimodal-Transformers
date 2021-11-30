from transformers import BertTokenizer, BertForMaskedLM
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import glob
import random
from transformers import BertConfig
from transformers import AdamW
import numpy as np
import shutil

import logging
import time
from datetime import timedelta

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _kl_divergence_with_logits(p_logits, q_logits):
    p = F.softmax(p_logits,dim=-1)
    log_p = F.log_softmax(p_logits,dim=-1)
    log_q = F.log_softmax(q_logits,dim=-1)

    kl = torch.sum(p * (log_p - log_q),-1)
    return kl

class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)
def create_logger(filepath):
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    

    return logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def save_checkpoint(state, is_best, checkpoint_path, filename="checkpoint.pt"):
    filename = os.path.join(checkpoint_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_path, "model_best.pt"))


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint["state_dict"])


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}
    
def mask(input_ids):
    output_tokens=input_ids.clone().detach()
    choises_idx=torch.randperm(256)
    num_mask=0
    max_mask=int(0.15*(input_ids.size()[0]))
    for index in choises_idx:
        if num_mask==max_mask:
            break
        # 80% of the time, replace with [MASK]
        if random.uniform(0, 1)< 0.8:
            masked_token = 3
        else:
            # 10% of the time, keep original
            if random.uniform(0, 1) < 0.5:
                masked_token = input_ids[index]
            # 10% of the time, replace with random word
            else:
                masked_token = random.randint(4,1027)

        output_tokens[index] = masked_token
        num_mask+=1
    labels = input_ids.masked_fill( output_tokens != 3, -100)
    return  output_tokens, labels

def prepo_data(data,data_dir):
    len_data=len(data)
    imgs_ids=[]
    for d in data:
        path=data_dir+str(d["id"])+".pt"
        img_ids= torch.load(path).long()
        imgs_ids.append(img_ids)
    tensor_ids=torch.stack(imgs_ids)+4
    tensor_ids_mask=[]
    labels_mask=[]
    for t in tensor_ids:
        im, lm= mask(t)
        tensor_ids_mask.append(im)
        labels_mask.append(lm)
    tensor_ids_mask=torch.stack(tensor_ids_mask)
    labels_mask=torch.stack(labels_mask)
    tensor_ids_mask=torch.cat([torch.ones((len_data, 1)).long(),tensor_ids_mask,2*torch.ones((len_data, 1)).long()],dim=1)
    labels_mask=torch.cat([-100*torch.ones((len_data, 1)).long(),labels_mask,-100*torch.ones((len_data, 1)).long()],dim=1)
    attmask=torch.ones(tensor_ids_mask.size()).long()
    encodings = {'input_ids': tensor_ids_mask, 'attention_mask': attmask, 'labels': labels_mask}
    dataset = Dataset(encodings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=38, shuffle=True)
    return loader

def prepo_data_aug(data,data_dir_ori,data_dir_aug):
    len_data=len(data)
    imgs_ids=[]
    imgs_ids_aug=[]
    for d in data:
        path=data_dir_ori+str(d["id"])+".pt"
        img_ids= torch.load(path).long()
        imgs_ids.append(img_ids)
        
        path_aug=data_dir_aug+str(d["id"])+".pt"
        img_ids_aug= torch.load(path_aug).long()
        imgs_ids_aug.append(img_ids_aug)
    tensor_ids=torch.stack(imgs_ids)+4
    tensor_ids_aug=torch.stack(imgs_ids_aug)+4
    encodings = {'input_ids':  tensor_ids, 'input_ids_aug': tensor_ids_aug}
    dataset = Dataset(encodings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=38, shuffle=True)
    return loader

def model_eval(data, model):
    with torch.no_grad():
        losses= []
        for batch in data:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
            loss = outputs.loss
            losses.append(loss.item())
            

    metrics = {"loss": np.mean(losses)}
    
    return metrics
if __name__ == "__main__":
    set_seed(1)
    train=[json.loads(l) for l in open("data_sets/proc/moviescope/train.jsonl")]
    val=[json.loads(l) for l in open("data_sets/proc/moviescope/dev.jsonl")]
    test=[json.loads(l) for l in open("data_sets/proc/moviescope/test.jsonl")]
    data_dir="data_sets/proc/moviescope/img_indices2/"
    data_dir_aug="data_sets/proc/moviescope/img_indices_aug/"
    
    uda=1
    
    train_loader=prepo_data(train,data_dir)
    if uda:
        train_loader_aug=prepo_data_aug(train,data_dir,data_dir_aug)
    val_loader=prepo_data(val,data_dir)
    test_loader=prepo_data(test,data_dir)
    
    config= BertConfig(vocab_size=1028,max_position_embeddings=258)
    model = BertForMaskedLM(config)
    model.cuda()
    
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    logger = create_logger("bert_img_uda/logfile.log")
    logger.info(model)
    
    if os.path.exists(os.path.join("bert_img_uda", "checkpoint.pt")):
        checkpoint = torch.load(os.path.join("bert_img_uda", "checkpoint.pt"))
        model.load_state_dict(checkpoint["state_dict"])
        
    epochs = 1000
    best_metric = np.inf
    gradient_accumulation_steps=256
    global_step=0
    len_train =len(train_loader)
    for epoch in range(epochs):
        train_losses = []
        model.train()
        optimizer.zero_grad()
        if uda:
            train_loader_t=zip(train_loader,train_loader_aug)
        else:
            train_loader_t=train_loader
        # setup loop with TQDM and dataloader
        loop = tqdm( train_loader_t, total=len_train)
        for batches in loop:
            if uda:
                batch=batches[0]
                batch_unsup=batches[1]
            else:
                batch=batches
                
            ##Calculate sup_loss    
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            # process
            outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
            # extract loss
            loss = outputs.loss
            
            ##unsup loss
            if uda:
                input_ids = batch_unsup['input_ids'].cuda()
                input_ids_aug = batch_unsup['input_ids_aug'].cuda()
                q_logits = model.bert(input_ids=input_ids_aug,output_hidden_states=True).last_hidden_state[:,0]
                model.eval()
                p_logits = model.bert(input_ids=input_ids,output_hidden_states=True).last_hidden_state[:,0]
                model.train()
                loss_aug = _kl_divergence_with_logits(p_logits, q_logits)
                avg_unsup_loss=torch.sum(loss_aug)
                loss+=avg_unsup_loss
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            train_losses.append(loss.item())
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            global_step += 1
            if global_step % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
            
        model.eval()
        metrics=model_eval(val_loader,model)
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        logger.info("Val Loss: {:.4f}".format(metrics['loss']))
        tuning_metric=metrics['loss']
        is_improvement = tuning_metric < best_metric
        if is_improvement:
            logger.info("Improvement: {:.4f}".format(tuning_metric))
            best_metric = tuning_metric
        save_checkpoint(
            {
                "state_dict": model.state_dict(),
            },
            is_improvement,
            'bert_img_uda',
        )

        
    load_checkpoint(model, os.path.join("bert_img_uda", "model_best.pt"))
    model.eval()
    test_metrics = model_eval(test_loader, model)
    logger.info("Test Loss: {:.4f}".format(test_metrics['loss']))
        
