import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import h5py

class ImagecapDataset(Dataset):

    def __init__(self, datasetname, split, img_transformer=None):
        '''
        :param datasetname: coco2014 or flickr30k
        :param split: train val test
        :param img_transformer: transform functions to transform the image
        '''
        if datasetname in ['coco2014','coco2017']:
            min_word_freq = 4
        else:
            min_word_freq = 3
        file_name = f'./dataset/{split}_imagecap_{datasetname}_5_cap_per_img_{min_word_freq}_min_word_freq.json'
        if not os.path.isfile(file_name):
            raise NotImplementedError(f'dataloader error: do not exist {file_name}')
        self.data = json.load(open(file_name, 'r'))
        if split =='train':
            random.shuffle(self.data)
        # else:
        #     self.data = self.data[:10]
        self.img_transformer = img_transformer
        self.split = split
    def __getitem__(self, i):
        """
        returns:
        img: the image convereted into a tensor of shape (batch_size,3, 256, 256)
        caption: the ground-truth caption of shape (batch_size, caption_length)
        caplen: the valid length (without padding) of the ground-truth caption of shape (batch_size,1)
        """
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        data_item = self.data[i]
        img_filepath = data_item['image_path']
        assert os.path.isfile(img_filepath)
        image_data = Image.open(img_filepath).convert('RGB')
        if self.img_transformer is not None:
            image_data = self.img_transformer(image_data)
        if self.split == 'train':
            caption = data_item['encoded_cap']
            caption_len = data_item['caption_len']
            all_captions = data_item['encoded_all_caps']
            caption = torch.LongTensor(caption)
            all_captions = torch.LongTensor(all_captions)
            # caption_len = torch.LongTensor(caption_len)
            return image_data, caption, all_captions, caption_len
        else:
            all_captions = data_item['encoded_all_caps']
            captions_len = data_item['caption_len']
            all_captions = torch.LongTensor(all_captions)
            # captions_len = torch.LongTensor(captions_len)
            return image_data, all_captions, captions_len, img_filepath.split('/')[-1]
    def __len__(self):
        return len(self.data)


class ImagecapDatasetFromFeature(Dataset):

    def __init__(self, datasetname, split, img_transformer=None):
        '''
        :param datasetname: coco2014 or flickr30k
        :param split: train val test
        :param img_transformer: transform functions to transform the image
        '''
        if datasetname in ['coco2014','coco2017', 'cocorobust']:
            min_word_freq = 4
        else:
            min_word_freq = 3
        file_name = f'./dataset/{split}_imagecap_{datasetname}_5_cap_per_img_{min_word_freq}_min_word_freq.json'
        if not os.path.isfile(file_name):
            raise NotImplementedError(f'dataloader error: do not exist {file_name}')
        self.data = json.load(open(file_name, 'r'))
        if split =='train':
            random.shuffle(self.data)
        # else:
        #     self.data = self.data[:10]
        self.img_transformer = img_transformer
        self.split = split
        self.data_feature_path = f'../dataset/{datasetname}_bu_features/{split}'
    def __getitem__(self, i):
        """
        returns:
        img: the image convereted into a tensor of shape (batch_size,3, 256, 256)
        caption: the ground-truth caption of shape (batch_size, caption_length)
        caplen: the valid length (without padding) of the ground-truth caption of shape (batch_size,1)
        """
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        data_item = self.data[i]
        img_filepath = data_item['image_path']
        img_filename = img_filepath.split('/')[-1]
        assert os.path.isfile(os.path.join(self.data_feature_path, img_filename + '.hdf5'))
        image_feature_file = h5py.File(os.path.join(self.data_feature_path, img_filename + '.hdf5'), mode='r')
        image_data = image_feature_file['image_features'][:]
        image_feature_file.close()
        if self.img_transformer is not None:
            image_data = self.img_transformer(image_data)
            # print(image_data.size())
            image_data = image_data.squeeze(0)
            if image_data.size(0)<36:
                padding = torch.zeros(36-image_data.size(0), 2048)
                image_data = torch.cat((image_data, padding), dim=0)
                assert image_data.size(0) == 36

            # print(image_data.size())
        if self.split == 'train':
            caption = data_item['encoded_cap']
            caption_len = data_item['caption_len']
            all_captions = data_item['encoded_all_caps']
            caption = torch.LongTensor(caption)
            all_captions = torch.LongTensor(all_captions)
            # caption_len = torch.LongTensor(caption_len)
            return image_data, caption, all_captions, caption_len
        else:
            all_captions = data_item['encoded_all_caps']
            captions_len = data_item['caption_len']
            all_captions = torch.LongTensor(all_captions)
            # captions_len = torch.LongTensor(captions_len)
            return image_data, all_captions, captions_len, img_filepath.split('/')[-1]
    def __len__(self):
        return len(self.data)


