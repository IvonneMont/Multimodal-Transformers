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
    def __init__(self, data_path, tokenizer, transforms, vocab, args):
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"] if args.model != "mmbt" else ["[SEP]"]
        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        self.max_seq_len = args.max_seq_len
        if args.model == "mmbt":
            self.max_seq_len -= args.num_image_embeds

        self.transforms = transforms

    def __len__(self):
        return len(self.data)
    
    def moviescope(self, index):
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
        poster =None
        
        #feature poster
        if self.args.model=="mult2":
            poster = torch.load(os.path.join(self.data_dir, 'PosterFeaturesVQGAN', f'{str(self.data[index]["id"])}.pt'))
          
            '''
            file = open(os.path.join(self.data_dir, 'PosterFeatures', f'{str(self.data[index]["id"])}.p'), 'rb')
            data = pickle.load(file, encoding='bytes')
            poster = torch.from_numpy(data).squeeze(0)
            '''
           
        
        #raw poster
        if self.args.model=="mmbt":
            poster = Image.open(
                os.path.join(self.data_dir, 'MatchedPosters',
                         f'{str(self.data[index]["id"])}.jpg')
            ).convert("RGB")
            poster = self.transforms(image)
        
        return sentence, segment, poster
    
    def mmimdb(self, index):
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
        poster =None
        
        #feature poster
        if self.args.model=="mult2":
            poster = torch.load(os.path.join(self.data_dir, 'PosterFeaturesVQGAN', f'{str(self.data[index]["id"])}.pt'))
        '''
        image = Image.open(
            os.path.join(self.data_dir, 'mmimdb/dataset/',
                         f'{str(self.data[index]["id"])}.jpeg')
        ).convert("RGB")
        image = self.transforms(image)
        ''''
        
        return sentence, segment, poster

    def __getitem__(self, index):
        if self.args.task == "moviescope":
            sentence, segment, image= self.moviescope(index)
            
        if self.args.task == "mmimdb":
            sentence, segment, image= self.mmimdb(index)
        
        
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
        
        if self.args.model == "mmbt":
            # The first SEP is part of Image Token.
            segment = segment[1:]
            sentence = sentence[1:]
            # The first segment (0) is of images.
            segment += 1
            
        
        if self.args.model == "mmbt" or self.args.model == "mult2":
            return sentence, segment, image, label
        else:
            return None
        
        
        
