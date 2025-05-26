from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy
from model.text_feature_extract import TextExtract
import numpy as np
import torchvision.transforms.functional as F
import pdb


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])
        num_unlab_imgs = len(self.un_annos)

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        table.add_row(['unlabel', -1, num_unlab_imgs, -1])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': tokens
        }

        return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 args,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.mask_ratio = args.m_ratio

        self.tokenizer = SimpleTokenizer()
        #self.TextExtract = TextExtract()
        mean = 0.44916429
        std = 0.26856974

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        
        if self.transform is not None:
            img0 = self.transform(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            img3 = self.transform(img)
            img4 = self.transform(img)
            img5 = self.transform(img)
        
        cap_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        caption_tokens = cap_tokens.clone()
        
        mlm_tokens1, mlm_labels1 = self._build_continue_masked_tokens_and_labels(cap_tokens.cpu().numpy())
        mlm_tokens2, mlm_labels2 = self._build_continue_masked_tokens_and_labels(cap_tokens.cpu().numpy())
        mlm_tokens3, mlm_labels3 = self._build_continue_masked_tokens_and_labels(cap_tokens.cpu().numpy())
        mlm_tokens4, mlm_labels4 = self._build_continue_masked_tokens_and_labels(cap_tokens.cpu().numpy())
        mlm_tokens5, mlm_labels5 = self._build_continue_masked_tokens_and_labels(cap_tokens.cpu().numpy())
        
        #mlm_tokens1, mlm_labels1, st1 = self._build_random_masked_tokens_and_labels(cap_tokens.cpu().numpy())
        #mlm_tokens2, mlm_labels2, st2 = self._build_random_masked_tokens_and_labels(cap_tokens.cpu().numpy())
        #mlm_tokens3, mlm_labels3, st3 = self._build_random_masked_tokens_and_labels(cap_tokens.cpu().numpy())
        #mlm_tokens4, mlm_labels4, st4 = self._build_random_masked_tokens_and_labels(cap_tokens.cpu().numpy())
        #mlm_tokens5, mlm_labels5, st5 = self._build_random_masked_tokens_and_labels(cap_tokens.cpu().numpy())

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img0,
            'images1': img1,
            'images2': img2,
            'images3': img3,
            'images4': img4,
            'images5': img5,
            'caption_ids': caption_tokens,
            'mlm_ids1': mlm_tokens1,
            'mlm_labels1': mlm_labels1,
            'mlm_ids2': mlm_tokens2,
            'mlm_labels2': mlm_labels2,
            'mlm_ids3': mlm_tokens3,
            'mlm_labels3': mlm_labels3,
            'mlm_ids4': mlm_tokens4,
            'mlm_labels4': mlm_labels4,
            'mlm_ids5': mlm_tokens5,
            'mlm_labels5': mlm_labels5
        }

        return ret
    
    def _build_continue_masked_tokens_and_labels(self, tokens):
        """
        Masking some comtinue tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        end = tokens.argmax()
        num = int(end*self.mask_ratio) #5    # 0.1    0.05
        st = random.randint(1, end-num-1)
        #pdb.set_trace()
        #labels = []
        labels = np.zeros_like(tokens)
        for i in range(num):
            labels[st+i] += tokens[st+i]
            #labels.append(tokens[st+i])
            tokens[st+i] = random.choice(token_range)

        return torch.tensor(tokens), torch.tensor(labels)

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels), 1