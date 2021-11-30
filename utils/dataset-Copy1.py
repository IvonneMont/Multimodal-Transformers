import json
import numpy as np
import os
from PIL import Image , ImageFile

import torchvision.transforms as transforms
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset

from utils.utils import truncate_seq_pair, numpy_seed


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args,uda=0):
        
        if args.model in ["FastText","MultFusion"]:
            embeddings_dict = {}
            with open(args.glove_path, 'r', encoding="utf-8") as f:
                i=0
                for line in f:
                    i=i+1
                    if i>3000:
                        break
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], "float32")
                    embeddings_dict[word] = torch.from_numpy(vector)
            self.embeddings_dict=embeddings_dict
            
            
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"] if args.model not in ["mmbt","mmbt_uda","mmbt2"] else ["[SEP]"]
        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        self.max_seq_len = args.max_seq_len
        if args.model in ["mmbt","mmbt_uda","mmbt2"]:
            self.max_seq_len -= args.num_image_embeds

        self.transforms = transforms

        self.name_p=os.path.split(data_path)[-1]
        self.uda=uda

    def __len__(self):
        return len(self.data)
    
    def moviescope(self, index):
        sentence, segment, poster, poster_aug = None, None, None, None
        
        # Load text or index images
        key_text="synopsis"
        
        if self.args.model in ["FastText","MultFusion"]:
            sentence = self.tokenizer(self.data[index][key_text])
            emb      = torch.zeros(300) 
            N=0
            for w in sentence:
                if w in self.vocab.stoi:
                    N=N+1
                    emb += self.embeddings_dict[w]
            if N==0:
                N=1
            emb=(1.0/N)*emb
            sentence= emb
        
        else:
            if self.args.model == "bert_img":
                sentence = torch.load(os.path.join(self.data_dir, 'img_indices2', f'{str(self.data[index]["id"])}.pt')).long()
                sentence = sentence[:(self.args.max_seq_len - 1)]
                segment = torch.zeros(len(sentence)) 
            elif (self.args.model != "imgbert"): 
                sentence = (
                    self.text_start_token
                    + self.tokenizer(self.data[index][key_text])[
                        : (self.args.max_seq_len - 1)
                    ]
                )
                segment = torch.zeros(len(sentence)) 

                sentence = torch.LongTensor(
                    [
                        self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                        for w in sentence
                    ]
                )
        # Load poster
        
        #feature poster
        if self.args.model in ["FastImage","mult2","MultFusion"]:
            if self.args.encoder=="VGG16":
                file = open(os.path.join(self.data_dir, 'PosterFeatures', f'{str(self.data[index]["id"])}.p'), 'rb')
                data = pickle.load(file, encoding='bytes')
                poster = torch.from_numpy(data).squeeze(0)
                
            if self.args.encoder=="VQGAN":
                if self.name_p=="train.jsonl":
                    poster = torch.load(os.path.join(self.data_dir, 'PosterFeaturesVQGAN_p', f'{str(self.data[index]["id"])}_a.pt'))
                else:
                    poster = torch.load(os.path.join(self.data_dir, 'PosterFeaturesVQGAN', f'{str(self.data[index]["id"])}.pt'))
           
        
        #raw poster
        if self.args.model in ["mmbt","mmbt_uda"]:
            if self.args.poster=="raw":
                #if self.name_p=="train.jsonl":
                if str(self.data[index]["id"]).find('a') != -1:
                    poster = Image.open(
                        os.path.join(self.data_dir, 'img',
                            f'{str(self.data[index]["id"])}.jpg')
                    ).convert("RGB")
                else:       
                    poster = Image.open(
                        os.path.join(self.data_dir, 'MatchedPosters',
                            f'{str(self.data[index]["id"])}.jpg')
                    ).convert("RGB")
                poster = self.transforms(poster)
            if self.args.poster=="features":
                if self.args.encoder=="VGG16":
                    file = open(os.path.join(self.data_dir, 'PosterFeatures', f'{str(self.data[index]["id"])}.p'), 'rb')
                    data = pickle.load(file, encoding='bytes')
                    poster = torch.from_numpy(data).squeeze(0)
                    poster = torch.reshape(poster,(1,poster.size()[0]))
                
                if self.args.encoder=="VQGAN":
                    poster = torch.load(os.path.join(self.data_dir, 'PosterFeaturesVQGAN', f'{str(self.data[index]["id"])}.pt'))
        if self.args.model == "mmbt_uda" and self.name_p in ["train.jsonl","train_500.jsonl","train_1000.jsonl","train_2000.jsonl"]:
            poster_aug = Image.open(
                        os.path.join(self.data_dir, 'img',
                            f'{str(self.data[index]["id"])}_a.jpg')
                    ).convert("RGB")
            poster_aug = self.transforms(poster_aug)
        
        if self.args.model in ["mmbt2","imgbert"]:
            poster = torch.load(os.path.join(self.data_dir, 'img_indices2', f'{str(self.data[index]["id"])}.pt')).long()
            poster =poster+1
            t=self.args.num_image_embeds-poster.size(dim=0)
           
            if t>0:
                poster=torch.cat([poster, torch.zeros(t).long()], dim=0)

            else:
                poster = poster[:self.args.num_image_embeds]
        
        return sentence, segment, poster, poster_aug
    
    def mmimdb(self, index):
        sentence, segment, poster, poster_aug = None, None, None, None
        
        key_text="synopsis"
        sentence = (
            self.text_start_token
            + self.tokenizer(self.data[index][key_text])[
                : (self.args.max_seq_len - 1)
            ]
        )
        segment = torch.zeros(len(sentence)) 

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )
        
        # Load poster
        
        #feature poster
        if self.args.model in ["FastImage","mult2","MultFusion"]:

            if self.args.encoder=="VQGAN":
                poster = torch.load(os.path.join(self.data_dir, 'PosterFeaturesVQGAN', f'{str(self.data[index]["id"])}.pt'))
           
        
        #raw poster
        if self.args.model in ["mmbt","mmbt_uda"]:
            if self.args.poster=="raw":
                if str(self.data[index]["id"]).find('a') != -1:
                    poster = Image.open(
                        os.path.join(self.data_dir, 'img_mmimdb',
                            f'{str(self.data[index]["id"])}.jpeg')
                    ).convert("RGB")
                else:       
                    poster = Image.open(
                        os.path.join(self.data_dir, 'mmimdb/dataset',
                            f'{str(self.data[index]["id"])}.jpeg')
                    ).convert("RGB")
                poster = self.transforms(poster)
              
            if self.args.poster=="features":
                if self.args.encoder=="VQGAN":
                    poster = torch.load(os.path.join(self.data_dir, 'PosterFeaturesVQGAN', f'{str(self.data[index]["id"])}.pt'))

        if self.args.model == "mmbt_uda" and self.name_p in ["train.jsonl","train_500.jsonl","train_1000.jsonl","train_2000.jsonl"]:
            poster_aug = Image.open(
                        os.path.join(self.data_dir, 'img_mmimdb',
                            f'{str(self.data[index]["id"])}_a.jpeg')
                    ).convert("RGB")
            poster_aug = self.transforms(poster_aug)
            
        return sentence, segment, poster, poster_aug
    

    def __getitem__(self, index):
        if self.args.task == "moviescope":
            sentence, segment, image, image_aug= self.moviescope(index)
            
        if self.args.task == "mmimdb":
            sentence, segment, image, image_aug= self.mmimdb(index)
        
        
        # Process labels
        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            label[
                [self.args.labels.index(tgt) for tgt in self.data[index]["label"]]
            ] = 1
        else:
            label = torch.LongTensor(
                [self.args.labels.index(self.data[index]["label"])]
            )
        
        if self.args.model in ["mmbt","mmbt_uda"]:
            # The first SEP is part of Image Token.
            segment = segment[1:]
            sentence = sentence[1:]
            # The first segment (0) is of images.
            segment += 1
        if image_aug != None and self.args.model=="mmbt_uda":
            if label[9]==1:
                image_aug=image
            
        
        return sentence, segment, image, image_aug, label, self.data[index]["id"]
        
        
        
        