import json
import csv
import torch
import numpy as np
import os
import skimage.transform
import random
from config import imgcap_adaptive_argument_parser, imgcap_gridTD_argument_parser, imgcap_aoa_argument_parser
from models import aoamodel
from models import gridTDmodel
from models import adaptiveattention
from nltk.corpus import stopwords
import glob
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import yaml
STOP_WORDS = list(set(stopwords.words('english')))
COCO_CATEGORY = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
                  'bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','hat',
                  'umbrella','shoe','handbag','tie','suitcase','frisbee', 'skis', 'snowboard', 'kite',
                  'skateboard','surfboard','bottle','plate','cup','fork','knife','spoon','bowl','banana','apple',
                  'sandwich','orange','broccoli','carrot','pizza','donut','cake','chair','couch','bed','mirror','window','desk',
                  'toilet','door','tv','laptop','mouse','remote','keyboard','microwave', 'oven', 'toaster','sink','refrigerator','blender',
                  'book','clock','vase','scissors','toothbrush', #73
                  'ball', 'bat', 'glove', 'racket', 'light', 'hydrant', 'sign', 'meter', 'glass','bear', 'drier', 'brush', 'plant', 'table', 'phone'] #total 88

PERSON = ['people', 'woman', 'women', 'man', 'men', 'boy', 'girl', 'player', 'baby', 'person']
AIRPLANE = ['plane', 'jetliner', 'jet', 'airplane']
BICYCLE = ['bike', 'bicycle']
CAR = ['car', 'taxi']

object_words_list = COCO_CATEGORY + PERSON + AIRPLANE + BICYCLE+ CAR
object_words_list = list(set(object_words_list))


flickr_frequent = ['dogs', 'building', 'person', 'background', 'field', 'women', 'hat', 'ball', 'children', 'child', 'water',
                   'street', 'boy', 'dog', 'girl', 'men', 'shirt',  'people', 'woman', 'man'] #20

coco_frequent =['clock', 'kitchen', 'picture',  'water', 'food', 'pizza', 'grass',  'building', 'bus', 'sign',
                'bathroom',  'baseball', 'dog', 'room', 'cat', 'plate', 'train',  'field',  'tennis', 'person', 'table', 'street', 'woman',  'people',  'man'] # 25


class EvaluationExperiments(object):
    def __init__(self, explainer):
        '''
        This class implement the abalation experiment for linguistic explanation and image explanation using MSCOCO dataset
        :param explainer: The explaination class
        :param data_file: the test json file with keys as image_path, encoded_all_caps, caption_len
        '''
        self.explainer = explainer
        self.explainer.model.eval()
        self.word_map = self.explainer.word_map
        self.rev_word_map = {v: k for k, v in self.word_map.items()}
        self.num_delete_patches= 20
        self.patch_size = 8
    def block_image(self, relevance):
        h, w = relevance.size()
        assert h % self.patch_size == 0
        assert w % self.patch_size == 0
        relevance = relevance.detach().cpu().numpy()
        # we generate a mask with square patches
        n_patch_h = h // self.patch_size  #28
        n_patch_w = w // self.patch_size  #28
        assert self.num_delete_patches <= n_patch_h * n_patch_w
        mask = np.arange(n_patch_h * n_patch_w)  # 28*28
        mask = mask.repeat(self.patch_size)  # 28*28*8
        mask = mask.reshape(n_patch_h, -1)  # (28, 28*8)
        mask = mask.repeat([self.patch_size], axis=0)
        patch_relevance = np.zeros(n_patch_h*n_patch_w)
        for idx in range(n_patch_w * n_patch_h):
            patch_relevance[idx] = np.sum(relevance[mask==idx])
        top_k_idx = np.argpartition(-patch_relevance, self.num_delete_patches)[:self.num_delete_patches]
        mask_return = np.ones_like(mask)
        for idx in top_k_idx:
            mask_return[mask==idx] = 0
        plt.imshow(mask_return)
        plt.show()
        mask_return = torch.from_numpy(mask_return).float().cuda()
        return mask_return

    def ablation_experiment(self,  data, explanation_type,  save_path_ablation, do_attention=False):
        '''
        words ablation:For words with index larger than 6, we first explain the target word and delete the top-3 relevant words
        For each word we delete the top-k relevant image patches
        image ablation: For words with index larger than 1 and in the category list, we explain the target word and set the top 20 relevant patches
        as the mean, and generate a new sentence. we check whether the target word is within the new sentence. if the target word is still predicted after ablation,
        we calculate the softmax predicted score to see if the model is less confident to generate the target word.
        :return:
        '''
        # self.stop_word_ablation_count = 0
        # self.category_ablation_count = 0
        self.stop_word_scores_diff = {}
        self.category_scores_diff = {}
        self.image_disappear_count = []
        self.image_category_score_diff = []
        if do_attention:
            self.stop_word_scores_diff_random = {}
            self.category_scores_diff_random = {}
            self.image_disappear_count_random = []
            self.image_category_score_diff_random = []
            self.image_disappear_count_att = []
            self.image_category_score_diff_att = []
        data_i = data
        img_filepath = data_i['image_path']
        img_filename = img_filepath.split('/')[-1]
        relevance_imgs, relevance_previous_words = self.explainer.explain_caption(img_filepath)
        beam_caption_encoded = self.explainer.beam_caption_encode # this is a list with the encoded label of the predicted caption with <start>
        predicted_scores = self.explainer.predictions  # a tensor of (cap_length, vocab_size)
        assert len(beam_caption_encoded) - 1 == len(relevance_previous_words)
        sentence_length = len(beam_caption_encoded) - 1  # the first element of beam_caption-encoded is <start>
        # print(sentence_length)
        assert len(relevance_imgs) == sentence_length
        '''============ablation==================='''
        with torch.no_grad():
            for t in range(sentence_length):
                word_t = beam_caption_encoded[t+1]
                word_str = self.rev_word_map[word_t]
                single_key_flag = word_str in object_words_list or word_str.rstrip('s') in object_words_list or word_str.rstrip('es') in object_words_list or word_str.rstrip('ies') + 'y'in object_words_list
                if t>=1 and single_key_flag:
                    # here we perform the image ablation experiment
                    original_w_score = torch.softmax(predicted_scores[t], dim=-1)[word_t]
                    image = self.explainer.img.detach().clone()
                    relevance_img = relevance_imgs[t].clone()  # (1,C, H, W)
                    if relevance_img.dim() == 2:
                        cam_size = int(np.sqrt(relevance_img.shape[-1]))
                        relevance_img = relevance_img[0].reshape(cam_size, cam_size)
                        scale = int(self.explainer.args.height // cam_size)
                        relevance_img = skimage.transform.pyramid_expand(relevance_img.detach().cpu().numpy(),
                                                                         upscale=scale,
                                                                         multichannel=False)
                        spatial_relevance = torch.from_numpy(relevance_img).cuda()
                    else:
                        spatial_relevance = torch.mean(relevance_img, dim=(0, 1))  # (H,W)
                    mask = self.block_image(spatial_relevance)
                    image_modified = mask * image
                    # plt.imshow(image_modified.permute(0,2,3,1).detach().cpu().numpy()[0])
                    # plt.show()
                    new_sentence, _ = self.explainer.model.beam_search(image_modified,self.explainer.word_map)
                    new_words = new_sentence[0].split()
                    print(new_words)
                    if word_str in new_words:
                        new_idx = new_words.index(word_str)
                        beam_caption_img = ['<start>']+new_words[:new_idx]
                        beam_caption_encoded_img = [self.word_map[w] for w in beam_caption_img]
                        # print(beam_caption_encoded_img)
                        new_predicted_scores = self.explainer.teacherforce_forward(image_modified, beam_caption_encoded_img)
                        assert new_predicted_scores.size(0) == new_idx + 1
                        new_w_img_score = torch.softmax(new_predicted_scores[-1], dim=-1)[word_t]
                        diff_img = original_w_score - new_w_img_score
                        self.image_category_score_diff.append([str(t), word_str, diff_img.item()])
                        print(word_t, word_str, t)
                        print('img_diff', diff_img.item(), original_w_score, new_w_img_score)
                    else:
                        self.image_disappear_count.append([str(t), word_str])
                        print('img_disappear', self.image_disappear_count)

                    if do_attention:
                        # here is the random
                        image = self.explainer.img.detach().clone()
                        relevance_img = relevance_imgs[t].clone()  # (1,C, H, W)
                        if relevance_img.dim() == 2:
                            cam_size = int(np.sqrt(relevance_img.shape[-1]))
                            relevance_img = relevance_img[0].reshape(cam_size, cam_size)
                            scale = int(self.explainer.args.height // cam_size)
                            relevance_img = skimage.transform.pyramid_expand(
                                relevance_img.detach().cpu().numpy(), upscale=scale,
                                multichannel=False)
                            spatial_relevance = torch.from_numpy(relevance_img).cuda()
                        else:
                            spatial_relevance = torch.mean(relevance_img, dim=(0, 1))  # (H,W)
                        h, w = spatial_relevance.size()
                        random_relevance = torch.tensor(random.sample(range(h * w), h * w))
                        random_relevance = random_relevance.view(h,w)
                        mask = self.block_image(random_relevance)
                        image_modified_random = mask * image
                        # plt.imshow(image_modified_random.permute(0,2,3,1).detach().cpu().numpy()[0])
                        # plt.show()
                        # caption_length_img = len(beam_caption_encoded[:t])
                        new_sentence_random, _ = self.explainer.model.beam_search(image_modified_random, self.explainer.word_map)
                        new_words_random = new_sentence_random[0].split()
                        print(new_words_random)
                        if word_str in new_words_random:
                            new_idx_random = new_words_random.index(word_str)
                            beam_caption_img_random = ['<start>'] + new_words_random[:new_idx_random]
                            beam_caption_encoded_img_random = [self.word_map[w] for w in beam_caption_img_random]
                            new_predicted_scores_random = self.explainer.teacherforce_forward(image_modified_random,
                                                                                       beam_caption_encoded_img_random)
                            assert new_predicted_scores_random.size(0) == new_idx_random + 1
                            new_w_img_score_random = torch.softmax(new_predicted_scores_random[-1], dim=-1)[word_t]
                            diff_img_random = original_w_score - new_w_img_score_random
                            self.image_category_score_diff_random.append([str(t), word_str,diff_img_random.item()])
                            print(word_t, word_str)
                            print('img_diff_random', diff_img_random.item(), original_w_score, new_w_img_score_random)
                        else:
                            self.image_disappear_count_random.append([str(t), word_str])
                            print('img_disappear_rdm', self.image_disappear_count_random)
                        #  the attention
                        image = self.explainer.img.detach().clone()
                        attention = self.explainer.alphas[t].detach().cpu().numpy()
                        if len(attention.shape) == 2:
                            attention = np.mean(attention, axis=0)
                        # print(attention.shape)
                        attention_size = int(np.sqrt(attention.shape[0]))
                        scale = int(self.explainer.args.height // attention_size)
                        attention = attention.reshape(attention_size, attention_size)
                        attention = skimage.transform.pyramid_expand(attention, upscale=scale,
                                                                     multichannel=False)
                        attention = self._project_maxabs(attention)
                        spatial_relevance = torch.from_numpy(attention).cuda()
                        mask = self.block_image(spatial_relevance)
                        image_modified_att = mask * image
                        # plt.imshow(image_modified_att.permute(0, 2, 3, 1).detach().cpu().numpy()[0])
                        # plt.show()
                        # caption_length_img = len(beam_caption_encoded[:t])
                        new_sentence_att, _ = self.explainer.model.beam_search(image_modified_att, self.explainer.word_map)
                        new_words_att = new_sentence_att[0].split()
                        print(new_words_att)
                        if word_str in new_words_att:
                            new_idx_att = new_words_att.index(word_str)
                            beam_caption_img_att = ['<start>'] + new_words_att[:new_idx_att]
                            beam_caption_encoded_img_att = [self.word_map[w] for w in beam_caption_img_att]
                            new_predicted_scores_att = self.explainer.teacherforce_forward(image_modified_att,
                                                                                       beam_caption_encoded_img_att)
                            assert new_predicted_scores_att.size(0) == new_idx_att + 1
                            new_w_img_score_att = torch.softmax(new_predicted_scores_att[-1], dim=-1)[word_t]
                            diff_img_att = original_w_score - new_w_img_score_att
                            self.image_category_score_diff_att.append([str(t), word_str,diff_img_att.item()])
                            print(word_t, word_str)
                            print('img_diff_att', diff_img_att.item(), original_w_score, new_w_img_score_att)
                        else:
                            self.image_disappear_count_att.append([str(t), word_str])
                            print('img_disappear_att', self.image_disappear_count_att)
                if t >= 6:
                    if word_str in STOP_WORDS or single_key_flag:
                        original_w_score = torch.softmax(predicted_scores[t], dim=-1)[word_t]
                        with torch.no_grad():
                            # here we perform the word ablation experiment
                            relevance_word = relevance_previous_words[t] #tensor with shape (t,)
                            # print(relevance_word.size())
                            assert relevance_word.size(0) == t+1
                            _, top_k_idx = torch.topk(relevance_word[1:], k=3) # do not consider the <start>
                            print(top_k_idx)
                            sub_caption = beam_caption_encoded[:t+1]
                            # print(sub_caption)
                            # print(top_k_idx)
                            deleted_sub_caption = list(np.delete(np.array(sub_caption),top_k_idx.cpu().numpy()+1)) # top_k +1 the retrive the index of the list including <start>
                            # print(deleted_sub_caption)
                            image = self.explainer.img.clone()
                            new_predicted_scores = self.explainer.teacherforce_forward(image, deleted_sub_caption)
                            assert new_predicted_scores.size(0) == len(deleted_sub_caption)
                            new_w_score = torch.softmax(new_predicted_scores[-1], dim=-1)[word_t]
                            diff = original_w_score - new_w_score

                            if word_str in STOP_WORDS:
                                if t not in self.stop_word_scores_diff:
                                    self.stop_word_scores_diff[t] = []
                                self.stop_word_scores_diff[t].append(diff.item())
                            else:
                                if t not in self.category_scores_diff:
                                    self.category_scores_diff[t] = []
                                self.category_scores_diff[t].append(diff.item())
                            print(word_t, word_str)
                            print('word', diff.item(), original_w_score, new_w_score)
                            if do_attention:
                                delete_id = random.sample(range(1,t), 3)
                                print(delete_id)
                                sub_caption_rdm = beam_caption_encoded[:t+1]
                                deleted_sub_caption_rdm = list(np.delete(np.array(sub_caption_rdm), delete_id))
                                # print(delete_id)
                                # print(sub_caption_rdm)
                                # print(deleted_sub_caption_rdm)
                                caption_length_rdm = len(deleted_sub_caption_rdm)
                                image = self.explainer.img.clone()
                                new_predicted_scores_rdm = self.explainer.teacherforce_forward(image, deleted_sub_caption_rdm)
                                assert new_predicted_scores_rdm.size(0) == caption_length_rdm
                                new_w_score_rdm = torch.softmax(new_predicted_scores_rdm[-1], dim=-1)[word_t]
                                diff_rdm = original_w_score - new_w_score_rdm
                                if word_str in STOP_WORDS:
                                    if t not in self.stop_word_scores_diff_random:
                                        self.stop_word_scores_diff_random[t] = []
                                    self.stop_word_scores_diff_random[t].append(diff_rdm.item())
                                else:
                                    if t not in self.category_scores_diff_random:
                                        self.category_scores_diff_random[t] = []
                                    self.category_scores_diff_random[t].append(diff_rdm.item())
                                print('word_rdm', diff_rdm.item(), original_w_score, new_w_score_rdm)
        ablation_results = []
        ablation_results.append({'words_ablation': [{'stop_words': self.stop_word_scores_diff}, {'category_words': self.category_scores_diff}],
                                 'image_ablation': [{'stop_words': self.image_disappear_count}, {'category_words': self.image_category_score_diff}]})
        print(ablation_results)
        with open(os.path.join(save_path_ablation, img_filename+'_'+explanation_type+'_ablation.json'), 'w') as f:
            json.dump(ablation_results, f)
        if do_attention:
            ablation_results_random = []
            ablation_results_random.append({'words_ablation': [{'stop_words': self.stop_word_scores_diff_random},
                                                        {'category_words': self.category_scores_diff_random}],
                                     'image_ablation': [{'stop_words': self.image_disappear_count_random},
                                                        {'category_words': self.image_category_score_diff_random}]})
            with open(os.path.join(save_path_ablation,img_filename+'_'+
                                   'random' + '_ablation.json'), 'w') as f:
                json.dump(ablation_results_random, f)

            ablation_results_att = []
            ablation_results_att.append({
                                     'image_ablation': [{'stop_words': self.image_disappear_count_att},
                                                        {'category_words': self.image_category_score_diff_att}]})
            with open(os.path.join(save_path_ablation,img_filename+'_'+
                                   'attention' + '_ablation.json'), 'w') as f:
                json.dump(ablation_results_att, f)
        torch.cuda.empty_cache()

    def _calculate_overlaped_pixels(self, bbox, relevance, threshold):

        '''threshold is a scalar between [0,1]'''
        # img = Image.fromarray(relevance * 255)
        # draw = ImageDraw.Draw(img)
        # draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline='red')
        # img.show()
        # print(bbox)
        bbox_mask = np.zeros(relevance.shape)
        bbox_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        # print(np.sum(bbox_mask>0))
        relevance_mask = relevance <= threshold
        if np.sum(relevance_mask>0):
            relevance[relevance_mask] = 0
        total_pixel_score = np.sum(relevance)
        if total_pixel_score == 0:
            return 0
        correct_pixel_score = np.sum(np.multiply(bbox_mask, relevance))
        ratio = 1.0 * correct_pixel_score / total_pixel_score
        if ratio > 1:
            return 1.
        else:
            return ratio
        # return correct_pixel_score

    def _project_maxabs(self, x):
        absmax = np.max(np.abs(x))
        if absmax == 0:
            return np.zeros(x.shape)
        x = 1.0 * x/absmax
        return x

    def bbox_experiment(self, category_dict, data, save_path_bbox, explanation_type='lrp', do_attention=False):
        data_i = data
        img_filepath = data_i['image_path']
        img_filename = img_filepath.split('/')[-1]

        relevance_imgs, relevance_previous_words = self.explainer.explain_caption(img_filepath)

        beam_caption_encoded = self.explainer.beam_caption_encode # this is a list with the encoded label of the predicted caption with <start>
        sentence_length = len(beam_caption_encoded) - 1

        explanation_correctness = {}
        explanation_correctness[img_filename] = {}
        if do_attention:
            attention_correctness = {}
            attention_correctness[img_filename] = {}

        category_data = category_dict[img_filename]
        categories = category_data['categories']
        # print(category_data)
        bboxes = category_data['bbox']
        resize_ratio = category_data['resize_ratio']
        # relevance_imgs, relevance_previous_words = self.explainer.explain_caption(img_filepath)
        # beam_caption_encoded = self.explainer.beam_caption_encode  # this is a list with the encoded label of the predicted caption with <start>
        # sentence_length = len(beam_caption_encoded) - 1
        with torch.no_grad():
            for t in range(sentence_length):
                word_t = beam_caption_encoded[t + 1]
                word_str = self.rev_word_map[word_t]

                for key in categories.keys():
                    single_key_flag = word_str == key or word_str.rstrip('s') == key or word_str.rstrip(
                        'es') == key or word_str.rstrip('ies') + 'y' == key
                    if len(key.split(' ')) > 1:
                        double_key_flag = word_str in key.split(' ') or word_str.rstrip('s') in key.split(
                            ' ') or word_str.rstrip('es') in key.split(' ') or word_str.rstrip(
                            'ies') + 'y' in key.split(
                            ' ')
                    else:
                        double_key_flag = False
                    if single_key_flag or double_key_flag:
                        relevance_img = relevance_imgs[t].detach().clone().cpu().numpy()
                        if key not in explanation_correctness[img_filename].keys():
                            explanation_correctness[img_filename][key] = {}
                            if do_attention:
                                attention_correctness[img_filename][key] = {}
                        if do_attention:
                            attention = self.explainer.alphas[t].detach().cpu().numpy()
                            if len(attention.shape) == 2:
                                attention = np.mean(attention, axis=0)
                            # print(attention.shape)
                            attention_size = int(np.sqrt(attention.shape[0]))
                            scale = int(self.explainer.args.height // attention_size)
                            attention = attention.reshape(attention_size, attention_size)
                            attention = skimage.transform.pyramid_expand(attention, upscale=scale,
                                                                         multichannel=False)
                            attention = self._project_maxabs(attention)
                        if explanation_type == 'GradCam':
                            cam_size = int(np.sqrt(relevance_img.shape[-1]))
                            relevance_img = relevance_img[0].reshape(cam_size, cam_size)
                            scale = int(self.explainer.args.height // cam_size)
                            relevance_img = skimage.transform.pyramid_expand(relevance_img, upscale=scale,
                                                                             multichannel=False)
                            relevance_img = self._project_maxabs(relevance_img)
                        elif 'neg' in explanation_type:
                            relevance_img = -1 * relevance_img
                            relevance_img = np.maximum(relevance_img, 0)
                            relevance_img = np.mean(relevance_img, axis=(0, 1))
                            relevance_img = self._project_maxabs(relevance_img)
                        else:
                            relevance_img = np.maximum(relevance_img, 0)
                            relevance_img = np.mean(relevance_img, axis=(0, 1))
                            relevance_img = self._project_maxabs(relevance_img)
                        # print(relevance_img.shape)
                        bbox = bboxes[categories[key]]
                        for box in bbox:
                            new_box = [0] * 4
                            new_box[0] = int(box[0] * resize_ratio[0])
                            new_box[1] = int(box[1] * resize_ratio[1])
                            new_box[2] = int(box[2] * resize_ratio[0])
                            new_box[3] = int(box[3] * resize_ratio[1])
                            for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                                if str(threshold) not in explanation_correctness[img_filename][key].keys():
                                    explanation_correctness[img_filename][key][str(threshold)] = 0
                                correct_score = self._calculate_overlaped_pixels(new_box, relevance_img, threshold)
                                if correct_score > explanation_correctness[img_filename][key][str(threshold)]:
                                    explanation_correctness[img_filename][key][str(threshold)] = correct_score
                                if do_attention:
                                    if str(threshold) not in attention_correctness[img_filename][key].keys():
                                        attention_correctness[img_filename][key][str(threshold)] = 0
                                    attention_score = self._calculate_overlaped_pixels(new_box, attention, threshold)
                                    if attention_score > attention_correctness[img_filename][key][str(threshold)]:
                                        attention_correctness[img_filename][key][str(threshold)] = attention_score
            new_predicted_scores_random = self.explainer.teacherforce_forward(self.explainer.img.detach().clone(),
                                                                              beam_caption_encoded)
            print(explanation_correctness[img_filename])
            if do_attention:
                print(attention_correctness[img_filename])
        with open(os.path.join(save_path_bbox, img_filename + '_' + explanation_type + 'correctness.json'), 'w') as f:
            json.dump(explanation_correctness, f)
        if do_attention:
            with open(os.path.join(save_path_bbox, img_filename + '_' + 'attention_correctness.json'), 'w') as f:
                json.dump(attention_correctness, f)

        torch.cuda.empty_cache()

    def tpfp_experiment(self,data, explanation_type,  save_path_tpfp, frequent_list, do_attention):
        quantile_point = [i/100 for i in range(0, 100)]
        self.TP_statistics = []
        self.FP_statistics = []
        self.TP_statistics_beta = []
        self.FP_statistics_beta = []
        if do_attention:
            self.TP_statistics_att = []
            self.FP_statistics_att = []


        data_i = data
        img_filepath = data_i['image_path']
        img_filename = img_filepath.split('/')[-1]
        ref_encoded_cap = data_i['encoded_all_caps']
        # build the vocabs in the reference captions, delete the start pad end and unk.
        ref_vocab = ref_encoded_cap[0]
        for c in range(1, len(ref_encoded_cap)):
            ref_vocab += ref_encoded_cap[c]
        ref_vocab = list(dict.fromkeys(ref_vocab))
        ref_vocab = [x for x in ref_vocab if x not in [self.word_map['<start>'], self.word_map['<pad>'], self.word_map['<end>'], self.word_map['<unk>']]]
        relevance_imgs, relevance_previous_words = self.explainer.explain_caption(img_filepath)
        beam_caption_encoded = self.explainer.beam_caption_encode # this is a list with the encoded label of the predicted caption with <start>
        assert len(beam_caption_encoded) - 1 == len(relevance_previous_words)
        sentence_length = len(beam_caption_encoded) - 1  # the first element of beam_caption-encoded is <start>
        # print(sentence_length)
        assert len(relevance_imgs) == sentence_length
        with torch.no_grad():
            for t in range(sentence_length):
                word_t = beam_caption_encoded[t+1]
                word_str = self.rev_word_map[word_t]
                if word_str in frequent_list and word_t in ref_vocab:
                    relevance_img_tp = relevance_imgs[t].detach().clone().cpu().numpy()
                    if do_attention:
                        attention = self.explainer.alphas[t]
                        if attention.dim() == 2:
                            attention = attention.mean(0)
                        attention = attention.detach().cpu().numpy()
                        attention_size = int(np.sqrt(attention.shape[0]))
                        scale = int(self.explainer.args.height // attention_size)
                        attention = attention.reshape(attention_size, attention_size)
                        attention = skimage.transform.pyramid_expand(attention, upscale=scale,
                                                                     multichannel=False)
                        attention_quantile = np.quantile(attention,quantile_point)
                        attention_quantile_list = [str(item) for item in attention_quantile]
                    # print(relevance_img_tp.shape)
                    if explanation_type == 'GradCam':
                        gradcam_size = int(np.sqrt(relevance_img_tp.shape[1]))
                        scale = int(self.explainer.args.height // gradcam_size)
                        relevance_img_tp = relevance_img_tp.squeeze().reshape(gradcam_size, gradcam_size)
                        relevance_img_tp = skimage.transform.pyramid_expand(relevance_img_tp, upscale=scale,
                                                                     multichannel=False)
                    else:
                        relevance_img_tp = np.mean(relevance_img_tp,axis=(0,1))  #(H, W)


                    if np.sum(relevance_img_tp>0) == 0:
                        mean_pos = 0
                    else:
                        mean_pos = np.sum(np.maximum(relevance_img_tp,0))/ np.sum(relevance_img_tp>0)
                    quantile = np.quantile(relevance_img_tp, quantile_point)
                    quantile_list = [str(item) for item in quantile]
                    self.TP_statistics.append({'word':word_str,'mean': str(np.mean(relevance_img_tp)), 'mean_abs': str(np.mean(np.abs(relevance_img_tp))),
                                               'mean_pos': str(mean_pos), 'max': str(np.max(relevance_img_tp)), 'quantile': quantile_list})
                    self.TP_statistics_beta.append({'word':word_str,'1-beta': str(1-self.explainer.betas[t].detach().cpu().item())})
                    if do_attention:
                        self.TP_statistics_att.append({'word':word_str,'mean': str(np.mean(attention)), 'max': str(np.max(attention)), 'quantile':attention_quantile_list})
                elif word_str in frequent_list and word_t not in ref_vocab:
                    relevance_img_fp = relevance_imgs[t].detach().clone().cpu().numpy()
                    if do_attention:
                        attention = self.explainer.alphas[t]
                        if attention.dim() == 2:
                            attention = attention.mean(0)
                        attention = attention.detach().cpu().numpy()
                        attention_size = int(np.sqrt(attention.shape[0]))
                        scale = int(self.explainer.args.height // attention_size)
                        attention = attention.reshape(attention_size, attention_size)
                        attention = skimage.transform.pyramid_expand(attention, upscale=scale,
                                                                     multichannel=False)
                        attention_quantile = np.quantile(attention, quantile_point)
                        attention_quantile_list = [str(item) for item in attention_quantile]
                    if explanation_type == 'GradCam':
                        gradcam_size = int(np.sqrt(relevance_img_fp.shape[1]))
                        scale = int(self.explainer.args.height // gradcam_size)
                        relevance_img_fp = relevance_img_fp.squeeze().reshape(gradcam_size, gradcam_size)
                        relevance_img_fp = skimage.transform.pyramid_expand(relevance_img_fp, upscale=scale,
                                                                            multichannel=False)
                    else:
                        relevance_img_fp = np.mean(relevance_img_fp, axis=(0, 1))  # (H, W)
                    if np.sum(relevance_img_fp>0) == 0:
                        mean_pos = 0
                    else:
                        mean_pos = np.sum(np.maximum(relevance_img_fp,0))/ np.sum(relevance_img_fp>0)
                    quantile = np.quantile(relevance_img_fp, quantile_point)
                    quantile_list = [str(item) for item in quantile]
                    self.FP_statistics.append({'word':word_str,'mean': str(np.mean(relevance_img_fp)), 'mean_abs': str(np.mean(np.abs(relevance_img_fp))),
                                               'mean_pos': str(mean_pos), 'max': str(np.max(relevance_img_fp)), 'quantile': quantile_list})
                    self.FP_statistics_beta.append({'word':word_str,'1-beta': str(1-self.explainer.betas[t].detach().cpu().item())})
                    if do_attention:
                        self.FP_statistics_att.append({'word':word_str,'mean': str(np.mean(attention)), 'max': str(np.max(attention)),'quantile':attention_quantile_list})
            new_predicted_scores_random = self.explainer.teacherforce_forward(self.explainer.img.detach().clone(),
                                                                          beam_caption_encoded)
        print(self.FP_statistics)
        print(self.TP_statistics)
        print(self.FP_statistics_beta)
        print(self.TP_statistics_beta)
        if do_attention:
            print(self.FP_statistics_att)
            print(self.TP_statistics_att)
        with open(os.path.join(save_path_tpfp,  img_filename+'_'+explanation_type+'_TP_statistics.json'), 'w') as f:
            json.dump(self.TP_statistics, f)
        with open(os.path.join(save_path_tpfp,  img_filename+'_'+explanation_type+'_FP_statistics.json'), 'w') as f:
            json.dump(self.FP_statistics, f)
        with open(os.path.join(save_path_tpfp,  img_filename +'_beta_FP_statistics.json'), 'w') as f:
            json.dump(self.FP_statistics_beta, f)
        with open(os.path.join(save_path_tpfp,  img_filename +'_beta_TP_statistics.json'), 'w') as f:
            json.dump(self.TP_statistics_beta, f)
        if do_attention:
            with open(os.path.join(save_path_tpfp, img_filename+'_'+
                                   'attention' + '_TP_statistics.json'), 'w') as f:
                json.dump(self.TP_statistics_att, f)
            with open(os.path.join(save_path_tpfp,img_filename+'_'+
                                   'attention' + '_FP_statistics.json'), 'w') as f:
                json.dump(self.FP_statistics_att, f)


class EvaluationExperimentsAOA(object):
    def __init__(self, explainer):
        '''
        This class implement the abalation experiment for linguistic explanation and image explanation using MSCOCO dataset
        :param explainer: The explaination class
        :param data_file: the test json file with keys as image_path, encoded_all_caps, caption_len
        '''
        self.explainer = explainer
        self.explainer.model.eval()
        self.word_map = self.explainer.word_map
        self.rev_word_map = {v: k for k, v in self.word_map.items()}
        self.num_delete_patches= 20
        self.patch_size = 8

    def _calculate_overlaped_pixels(self, bbox, relevance, threshold):

        '''threshold is a scalar between [0,1]'''
        # img = Image.fromarray(relevance * 255)
        # draw = ImageDraw.Draw(img)
        # draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline='red')
        # img.show()
        # print(bbox)
        bbox_mask = np.zeros(relevance.shape)
        bbox_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        relevance_mask = relevance <= threshold
        if np.sum(relevance_mask>0):
            relevance[relevance_mask] = 0
        total_pixel_score = np.sum(relevance)
        if total_pixel_score == 0:
            return 0
        correct_pixel_score = np.sum(np.multiply(bbox_mask, relevance))
        ratio = 1.0 * correct_pixel_score / total_pixel_score
        if ratio > 1:
            return 1.
        else:
            return ratio

    def _project_maxabs(self, x):
        absmax = np.max(np.abs(x))
        if absmax == 0:
            return np.zeros(x.shape)
        x = 1.0 * x/absmax
        return x

    def bbox_experiment(self, category_dict, data, save_path_bbox, explanation_type='lrp', head_idx=0, do_attention=False):
        explanation_correctness = {}
        if do_attention:
            attention_correctness = {}
        data_i = data
        img_filepath = data_i['image_path']
        img_filename = img_filepath.split('/')[-1]
        if os.path.isfile(os.path.join(save_path_bbox, img_filename + '_' + str(head_idx) + explanation_type + 'correctness.json')) and \
            os.path.isfile(os.path.join(save_path_bbox, img_filename + explanation_type + 'correctness.json')) and do_attention:
            raise Warning('the file already exists')
        if os.path.isfile(os.path.join(save_path_bbox, img_filename + '_' + str(head_idx) + explanation_type + 'correctness.json')) and do_attention==False:
            raise Warning('the file already exists')
        category_data = category_dict[img_filename]
        categories = category_data['categories']
        bboxes = category_data['bbox']
        resize_ratio = category_data['resize_ratio']
        if explanation_type == 'lrp':
            relevance_imgs, relevance_previous_words = self.explainer.explain_caption(img_filepath, head_idx)
        else:
            relevance_imgs, relevance_previous_words = self.explainer.explain_caption(img_filepath)
        with torch.no_grad():
            beam_caption_encoded = self.explainer.beam_caption_encode  # this is a list with the encoded label of the predicted caption with <start>
            sentence_length = len(beam_caption_encoded) - 1
            for t in range(sentence_length):
                word_t = beam_caption_encoded[t + 1]
                word_str = self.rev_word_map[word_t]
                for key in categories.keys():
                    single_key_flag = word_str == key or word_str.rstrip('s') == key or word_str.rstrip('es') == key or word_str.rstrip('ies')+'y' == key
                    if len(key.split(' ')) > 1:
                        double_key_flag = word_str in key.split(' ') or word_str.rstrip('s') in key.split(' ') or word_str.rstrip('es') in key.split(' ') or word_str.rstrip('ies')+'y' in key.split(' ')
                    else:
                        double_key_flag = False
                    if single_key_flag or double_key_flag:
                        relevance_img = relevance_imgs[t].detach().clone().cpu().numpy()
                        if key not in explanation_correctness.keys():
                            explanation_correctness[key] = {}
                            if do_attention:
                                attention_correctness[key] = {}
                        if do_attention:
                            attention = self.explainer.alphas[t][head_idx].detach().cpu().numpy()
                            attention_size = int(np.sqrt(attention.shape[0]))
                            scale = int(self.explainer.args.height // attention_size)
                            attention = attention.reshape(attention_size, attention_size)
                            attention = skimage.transform.pyramid_expand(attention, upscale=scale,
                                                                         multichannel=False)
                            attention = self._project_maxabs(attention)
                        if explanation_type == 'GradCam':
                            cam_size = int(np.sqrt(relevance_img.shape[-1]))
                            relevance_img = relevance_img[0].reshape(cam_size, cam_size)
                            scale = int(self.explainer.args.height // cam_size)
                            relevance_img = skimage.transform.pyramid_expand(relevance_img, upscale=scale,
                                                                             multichannel=False)
                            relevance_img = self._project_maxabs(relevance_img)
                        elif 'neg' in explanation_type:
                            relevance_img = -1 * relevance_img
                            relevance_img = np.maximum(relevance_img, 0)
                            relevance_img = np.mean(relevance_img, axis=(0, 1))
                            relevance_img = self._project_maxabs(relevance_img)
                        else:
                            relevance_img = np.maximum(relevance_img, 0)
                            relevance_img = np.mean(relevance_img, axis=(0, 1))
                            relevance_img = self._project_maxabs(relevance_img)
                        # print(relevance_img.shape)
                        bbox = bboxes[categories[key]]
                        for box in bbox:
                            new_box = [0] * 4
                            new_box[0] = int(box[0] * resize_ratio[0])
                            new_box[1] = int(box[1] * resize_ratio[1])
                            new_box[2] = int(box[2] * resize_ratio[0])
                            new_box[3] = int(box[3] * resize_ratio[1])
                            for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                                if str(threshold) not in explanation_correctness[key].keys():
                                    explanation_correctness[key][str(threshold)] = 0
                                correct_score = self._calculate_overlaped_pixels(new_box, relevance_img, threshold)
                                if correct_score > explanation_correctness[key][str(threshold)]:
                                    explanation_correctness[key][str(threshold)] = correct_score
                                if do_attention:
                                    if str(threshold) not in attention_correctness[key].keys():
                                        attention_correctness[key][str(threshold)] = 0
                                    attention_score = self._calculate_overlaped_pixels(new_box, attention, threshold)
                                    if attention_score > attention_correctness[key][str(threshold)]:
                                        attention_correctness[key][str(threshold)] = attention_score
            new_predicted_scores= self.explainer.teacherforce_forward(self.explainer.img.detach().clone(), beam_caption_encoded)
        print('relevance',explanation_correctness)
        if do_attention:
            print('attention',attention_correctness)
        if explanation_type == 'lrp':
            with open(os.path.join(save_path_bbox, img_filename + '_' + str(head_idx) + explanation_type + 'correctness.json'), 'w') as f:
                json.dump(explanation_correctness, f)
            if do_attention:
                with open(os.path.join(save_path_bbox, img_filename + '_' + str(head_idx) + 'attention_correctness.json'), 'w') as f:
                    json.dump(attention_correctness, f)
        else:
            with open(os.path.join(save_path_bbox, img_filename + explanation_type + 'correctness.json'), 'w') as f:
                json.dump(explanation_correctness, f)
            if do_attention:
                with open(os.path.join(save_path_bbox, img_filename  + 'attention_correctness.json'), 'w') as f:
                    json.dump(attention_correctness, f)
        torch.cuda.empty_cache()

    def bbox_experiment_attention(self, category_dict, data, save_path_bbox, head_idx):
        attention_correctness = {}
        data_i = data
        img_filepath = data_i['image_path']
        img_filename = img_filepath.split('/')[-1]
        category_data = category_dict[img_filename]
        categories = category_data['categories']
        print(categories.keys())
        bboxes = category_data['bbox']
        resize_ratio = category_data['resize_ratio']
        with torch.no_grad():
            self.explainer.get_hidden_parameters(img_filepath)
            beam_caption_encoded = self.explainer.beam_caption_encode  # this is a list with the encoded label of the predicted caption with <start>
            sentence_length = len(beam_caption_encoded) - 1
            for t in range(sentence_length):
                word_t = beam_caption_encoded[t + 1]
                word_str = self.rev_word_map[word_t]
                for key in categories.keys():
                    single_key_flag = word_str == key or word_str.rstrip('s') == key or word_str.rstrip('es') == key or word_str.rstrip('ies')+'y' == key
                    if len(key.split(' ')) > 1:
                        double_key_flag = word_str in key.split(' ') or word_str.rstrip('s') in key.split(' ') or word_str.rstrip('es') in key.split(' ') or word_str.rstrip('ies')+'y' in key.split(' ')
                    else:
                        double_key_flag = False
                    if single_key_flag or double_key_flag:
                        attention = self.explainer.alphas[t][head_idx].detach().cpu().numpy()
                        attention_size = int(np.sqrt(attention.shape[0]))
                        scale = int(self.explainer.args.height // attention_size)
                        attention = attention.reshape(attention_size, attention_size)
                        attention = skimage.transform.pyramid_expand(attention, upscale=scale,
                                                                     multichannel=False)
                        attention = self._project_maxabs(attention)
                        if key not in attention_correctness.keys():
                            attention_correctness[key] = {}

                        bbox = bboxes[categories[key]]
                        for box in bbox:
                            new_box = [0] * 4
                            new_box[0] = int(box[0] * resize_ratio[0])
                            new_box[1] = int(box[1] * resize_ratio[1])
                            new_box[2] = int(box[2] * resize_ratio[0])
                            new_box[3] = int(box[3] * resize_ratio[1])
                            for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                                if str(threshold) not in attention_correctness[key].keys():
                                    attention_correctness[key][str(threshold)] = 0
                                attention_score = self._calculate_overlaped_pixels(new_box, attention, threshold)
                                if attention_score > attention_correctness[key][str(threshold)]:
                                    attention_correctness[key][str(threshold)] = attention_score
            new_predicted_scores= self.explainer.teacherforce_forward(self.explainer.img.detach().clone(), beam_caption_encoded)
            print('attention',attention_correctness)
            with open(os.path.join(save_path_bbox, img_filename + '_' + str(head_idx) + 'attention_correctness.json'), 'w') as f:
                json.dump(attention_correctness, f)
            torch.cuda.empty_cache()



def generate_evaluation_files(model_type='gridTD', explainer_type='lrp', head_idx=None, dataset='coco2017', do_attention=True):

    if model_type == 'gridTD':
        parser = imgcap_gridTD_argument_parser()
        args = parser.parse_args()
        if dataset == 'coco2017':
            args.weight = glob.glob('./output/gridTD/vgg16/coco2017/BEST_checkpoint_coco2017_epoch22*')[0]  # 22
        else:
            args.weight = glob.glob('./output/gridTD/vgg16/flickr30k/BEST_checkpoint_flickr30k_epoch28*')[0]
        args.dataset = dataset
        word_map_path = f'./dataset/wordmap_{args.dataset}.json'
        word_map = json.load(open(word_map_path, 'r'))
    elif model_type == 'aoa':
        parser = imgcap_aoa_argument_parser()
        args = parser.parse_args()
        if dataset == 'coco2017':
            args.weight = glob.glob('./output/aoa/vgg16/coco2017/BEST_checkpoint_coco2017_epoch34*')[0]
        else:
            args.weight = glob.glob('./output/aoa/vgg16/flickr30k/BEST_checkpoint_flickr30k_epoch31*')[0]
        args.dataset = dataset
        word_map_path = f'./dataset/wordmap_{args.dataset}.json'
        word_map = json.load(open(word_map_path, 'r'))

    else:
        raise NotImplementedError('model_type is aoa or gridTD')
    if dataset == 'coco2017':
        data_file = json.load(open('./dataset/test_imagecap_coco2017_5_cap_per_img_4_min_word_freq.json', 'r'))
        category_dict = json.load(open('./dataset/COCOvalEntities.json'))
    else:
        data_file = json.load(open('./dataset/test_imagecap_flickr30k_5_cap_per_img_3_min_word_freq.json', 'r'))

    for i in range(len(data_file)):
        # print(i)
        img_filename = data_file[i]['image_path'].split('/')[-1]
        if img_filename != '000000015746.jpg':
            continue
        if model_type == 'aoa':
            if 'lrp' in explainer_type:
                explainer = aoamodel.ExplainAOAAttention(args, word_map)
            elif 'GuidedGradCam' in explainer_type:
                explainer = aoamodel.ExplainAOAGuidedGradCam(args, word_map)
            elif explainer_type == 'GradCam':
                explainer = aoamodel.ExplainAOAGradCam(args, word_map)
            elif explainer_type == 'GuidedBackpropagate':
                explainer = aoamodel.ExplainAOAGuidedGradient(args, word_map)
            elif explainer_type == 'gradient':
                explainer = aoamodel.ExplainAOAGradient(args, word_map)
            else:
                raise NotImplementedError('no such explainer_type')
            evaluation_engin = EvaluationExperimentsAOA(explainer=explainer)
        elif model_type == 'gridTD':
            if 'lrp' in explainer_type:
                explainer = gridTDmodel.ExplainGridTDAttention(args, word_map)
            elif 'GuidedGradCam' in explainer_type:
                explainer =  gridTDmodel.ExplainGridTDGuidedGradCam(args, word_map)
            elif explainer_type == 'GradCam':
                explainer = gridTDmodel.ExplainGridTDGradCam(args, word_map)
            elif explainer_type == 'GuidedBackpropagate':
                explainer = gridTDmodel.ExplainiGridTDGuidedGradient(args, word_map)
            elif explainer_type == 'gradient':
                explainer = gridTDmodel.ExplainGridTDGradient(args, word_map)
            else:
                raise NotImplementedError('no such explainer_type')
            evaluation_engin = EvaluationExperiments(explainer=explainer)
        else:
            raise NotImplementedError('no such model type')
        explanation_type = explainer_type
        save_path_bbox = os.path.join(args.save_path, args.encoder, args.dataset, 'evaluation/bbox/', explanation_type)
        save_path_ablation = os.path.join(args.save_path, args.encoder, args.dataset, 'evaluation/ablation/', explanation_type)
        save_path_tpfp = os.path.join(args.save_path, args.encoder, args.dataset, 'evaluation/tpfp/', explanation_type)
        if not os.path.isdir(save_path_ablation):
            os.makedirs(save_path_ablation)
        if not os.path.isdir(save_path_bbox):
            os.makedirs(save_path_bbox)
        if not os.path.isdir(save_path_tpfp):
            os.makedirs(save_path_tpfp)
        evaluation_engin.ablation_experiment(data_file[i],explanation_type, save_path_ablation,do_attention=do_attention)
        evaluation_engin.explainer.model.zero_grad()
        if img_filename in category_dict:
            if model_type == 'aoa':
                evaluation_engin.bbox_experiment(category_dict, data_file[i], save_path_bbox, head_idx=head_idx)
            else:
                evaluation_engin.bbox_experiment(category_dict, data_file[i], save_path_bbox, explanation_type, do_attention=do_attention)
        evaluation_engin.explainer.model.zero_grad()
        evaluation_engin.tpfp_experiment(data_file[i], explanation_type,  save_path_tpfp, flickr_frequent, do_attention=do_attention)
        evaluation_engin.explainer.model.zero_grad()
        del evaluation_engin.explainer.model
        del evaluation_engin.explainer
        del evaluation_engin
        del explainer
        gc.collect()
        torch.cuda.empty_cache()


def analyze_bbox(model_type):
    if model_type == 'gridTD':
        parser = imgcap_gridTD_argument_parser()
        args = parser.parse_args()
    elif model_type == 'aoa':
        parser = imgcap_aoa_argument_parser()
        args = parser.parse_args()
    else:
        raise NotImplementedError('model_type in aoa, gridTD or adaptive')
    correctness = np.zeros((8, 10))
    for i, explanation_type in enumerate(['lrp', 'GuidedBackpropagate', 'gradient', 'GuidedGradCam', 'GradCam', 'attention', 'lrpneg', 'GuidedGradCamneg']):
        count = 0
        correctness_sub = np.zeros(10)
        path_bbox = os.path.join(args.save_path, args.encoder, 'coco2017/evaluation/bbox/', explanation_type)
        if explanation_type == 'attention':
            files = glob.glob(os.path.join(args.save_path, args.encoder, 'coco2017/evaluation/bbox/lrp/', '*'+explanation_type+'_correctness.json'))
        else:
            files = glob.glob(os.path.join(path_bbox,'*'+explanation_type+'correctness.json'))
        for file in files:
            data = json.load(open(file, 'r'))
            # print(data)
            for key in data:
                value = data[key]
                if value == {}:
                    continue
                for category in value:
                    item = value[category]
                    count += 1
                    for idx, th in enumerate(list(item.keys())):
                        correctness_sub[idx] += float(item[th])
        correctness_sub = correctness_sub / count
        correctness[i] = correctness_sub
    print(count)  #4691
    row_list = [['th']+[str(i/10) for i in range(10)]]
    for i, explanation_type in enumerate(['lrp', 'GuidedBackpropagate', 'gradient', 'GuidedGradCam', 'GradCam', 'attention','lrpneg', 'GuidedGradCamneg']):
        row_list.append([explanation_type]+correctness[i].astype(str).tolist())
    print(row_list)
    np.savetxt(os.path.join(args.save_path, args.encoder, 'coco2017/evaluation/bbox/', model_type+'_correctness.csv'), np.array(row_list), delimiter=',',fmt='%s')
    # with open(os.path.join(args.save_path, args.encoder, 'coco2017/evaluation/bboxcider/', 'correctness.csv'), 'w',newline="") as f:
    #     writer = csv.writer(f)
    #     for row in row_list:
    #         writer.writerow(row)


def analyze_ablation(model_type):
    if model_type == 'gridTD':
        parser = imgcap_gridTD_argument_parser()
        args = parser.parse_args()
    elif model_type == 'aoa':
        parser = imgcap_aoa_argument_parser()
        args = parser.parse_args()
    else:
        raise NotImplementedError('model_type in aoa, gridTD or adaptive')
    row_list = []
    for i, explanation_type in enumerate(['lrp', 'GuidedBackpropagate', 'gradient', 'GuidedGradCam', 'GradCam', 'attention', 'random']):
        count_word_stop_pos = 0
        score_word_stop_pos = 0
        count_word_cat_pos = 0
        score_word_cat_pos = 0

        count_word_stop_neg = 0
        score_word_stop_neg = 0
        count_word_cat_neg = 0
        score_word_cat_neg = 0
        count_img_disappear = 0
        count_img_cat_pos = 0
        count_img_cat_neg = 0
        score_img_cat_neg = 0
        score_img_cat_pos = 0

        path_ablation = os.path.join(args.save_path, args.encoder, 'coco2017/evaluation/ablation/', explanation_type)
        if explanation_type == 'attention':
            files = glob.glob(os.path.join(args.save_path, args.encoder, 'coco2017/evaluation/ablation/lrp/', '*'+explanation_type+'_ablation.json'))
        elif explanation_type == 'random':
            files = glob.glob(os.path.join(args.save_path, args.encoder, 'coco2017/evaluation/ablation/lrp/',
                                           '*' + explanation_type + '_ablation.json'))
        else:
            files = glob.glob(os.path.join(path_ablation,'*'+explanation_type+'_ablation.json'))
        for file in files:
            data = json.load(open(file, 'r'))
            if explanation_type != 'attention':
                word_stop = data[0]['words_ablation'][0]['stop_words']
                word_cat = data[0]['words_ablation'][1]['category_words']
            else:
                word_stop, word_cat = [], []
            img_stop = data[0]['image_ablation'][0]['stop_words']
            img_cat = data[0]['image_ablation'][1]['category_words']
            if len(word_stop) > 0:
                for key in word_stop.keys():
                    value = word_stop[key][0]
                    if value >=0:
                        count_word_stop_pos += 1
                        score_word_stop_pos += value
                    else:
                        count_word_stop_neg += 1
                        score_word_stop_neg += value
            if len(word_cat) > 0:
                for key in word_cat.keys():
                    value = word_cat[key][0]
                    if value >=0:
                        count_word_cat_pos += 1
                        score_word_cat_pos += value
                    else:
                        count_word_cat_neg += 1
                        score_word_cat_neg += value
            if len(img_stop) > 0:
                for item in img_stop:
                    count_img_disappear += 1
            if len(img_cat) > 0:
                for item in img_cat:
                    value = item[2]
                    if value >=0:
                        count_img_cat_pos += 1
                        score_img_cat_pos += value
                    else:
                        count_img_cat_neg += 1
                        score_img_cat_neg += value
        total_count = np.array([count_word_stop_neg + count_word_stop_pos,
                       count_word_cat_pos + count_word_cat_neg,
                       count_img_disappear + count_img_cat_pos + count_img_cat_neg,
                       count_img_disappear + count_img_cat_pos + count_img_cat_neg])
        count_pos = np.array([count_word_stop_pos, count_word_cat_pos, count_img_cat_pos,count_img_disappear,])
        score = np.array([score_word_stop_pos +score_word_stop_neg, score_word_cat_pos + score_word_cat_neg, score_img_cat_pos+score_img_cat_neg])
        score_pos = np.array([score_word_stop_pos, score_word_cat_pos, score_img_cat_pos])
        for i in range(3):
            if total_count[i] > 0:
                score[i] = score[i] / total_count[i]
                # score_pos[i] = score_pos[i] / count_pos[i]
        row_list.append(['total_num'] + total_count.tolist())
        count_pos = count_pos / total_count
        row_list.append([explanation_type] + count_pos.tolist())
        row_list.append([explanation_type] + score.tolist())
        # row_list.append([explanation_type] + score_pos.tolist())

        print(row_list)
    with open(os.path.join(args.save_path, args.encoder, 'coco2017/evaluation/ablation/', model_type+'_ablation.csv'), 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(row_list)


def analyze_ablation_aoa():

    parser = imgcap_aoa_argument_parser()
    args = parser.parse_args()

    row_list = []
    for i, explanation_type in enumerate(['lrp', 'GuidedBackpropagate', 'gradient', 'GuidedGradCam', 'GradCam', 'random']):
        count_word_stop_pos = 0
        score_word_stop_pos = 0
        count_word_cat_pos = 0
        score_word_cat_pos = 0

        count_word_stop_neg = 0
        score_word_stop_neg = 0
        count_word_cat_neg = 0
        score_word_cat_neg = 0
        path_ablation = os.path.join(args.save_path, args.encoder, 'coco2017/evaluation/ablation/', explanation_type)

        if explanation_type == 'random':
            files = glob.glob(os.path.join(args.save_path, args.encoder, 'coco2017/evaluation/ablation/lrp/',
                                           '*' + explanation_type + '_ablation.json'))
        else:
            files = glob.glob(os.path.join(path_ablation,'*'+explanation_type+'_ablation.json'))
        for file in files:
            data = json.load(open(file, 'r'))
            if explanation_type != 'attention':
                word_stop = data[0]['words_ablation'][0]['stop_words']
                word_cat = data[0]['words_ablation'][1]['category_words']
            else:
                word_stop, word_cat = [], []
            if len(word_stop) > 0:
                for key in word_stop.keys():
                    value = word_stop[key][0]
                    if value >=0:
                        count_word_stop_pos += 1
                        score_word_stop_pos += value
                    else:
                        count_word_stop_neg += 1
                        score_word_stop_neg += value
            if len(word_cat) > 0:
                for key in word_cat.keys():
                    value = word_cat[key][0]
                    if value >=0:
                        count_word_cat_pos += 1
                        score_word_cat_pos += value
                    else:
                        count_word_cat_neg += 1
                        score_word_cat_neg += value
        total_count = np.array([count_word_stop_neg + count_word_stop_pos,
                       count_word_cat_pos + count_word_cat_neg])
        count_pos = np.array([count_word_stop_pos, count_word_cat_pos])
        score = np.array([score_word_stop_pos +score_word_stop_neg, score_word_cat_pos + score_word_cat_neg])
        score_pos = np.array([score_word_stop_pos, score_word_cat_pos])
        for i in range(2):
            if total_count[i] > 0:
                score[i] = score[i] / total_count[i]
                # score_pos[i] = score_pos[i] / count_pos[i]
        count_pos = count_pos / total_count
        row_list.append(['total_num'] + total_count.tolist())
        row_list.append([explanation_type] + count_pos.tolist())
        row_list.append([explanation_type] + score.tolist())
        # row_list.append([explanation_type] + score_pos.tolist())

        print(row_list)
    with open(os.path.join(args.save_path, args.encoder, 'coco2017/evaluation/ablation/', 'aoa_ablation.csv'), 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(row_list)


def process_multihead_attention_bbox_aoa():
    test_data = json.load(open('./dataset/test_imagecap_coco2017_5_cap_per_img_4_min_word_freq.json', 'r'))
    category_dict = json.load(open('./dataset/COCOvalEntities.json'))
    for explanation_type in ['GradCam']:
        if not os.path.isdir(f'./output/aoa/vgg16/coco2017/evaluation/bbox/{explanation_type}_merge/'):
            os.makedirs(f'./output/aoa/vgg16/coco2017/evaluation/bbox/{explanation_type}_merge/')
        for data in test_data:
            img_filename = data['image_path'].split('/')[-1]
            if img_filename not in category_dict.keys():
                continue
            max_correctness_score = {}
            print(img_filename)
            for h in range(8):
                correctness_file_name = f'./output/aoa/vgg16/coco2017/evaluation/bbox/{explanation_type}/{img_filename}_{h}{explanation_type}correctness.json'
                if not os.path.isfile(correctness_file_name):
                    break
                correctness_file = json.load(open(correctness_file_name, 'r'))
                if correctness_file == {}:
                    json.dump(max_correctness_score, open(f'./output/aoa/vgg16/coco2017/evaluation/bbox/{explanation_type}_merge/{img_filename}_{explanation_type}correctness.json', 'w'))
                    break
                else:
                    for key in correctness_file.keys():
                        if key not in max_correctness_score.keys():
                            max_correctness_score[key] = {}
                        for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                            if str(threshold) not in max_correctness_score[key].keys():
                                max_correctness_score[key][str(threshold)] = 0.0
                            if correctness_file[key][str(threshold)] > max_correctness_score[key][str(threshold)]:
                                max_correctness_score[key][str(threshold)] = correctness_file[key][str(threshold)]
            json.dump(max_correctness_score, open(f'./output/aoa/vgg16/coco2017/evaluation/bbox/{explanation_type}_merge/{img_filename}_{explanation_type}correctness.json', 'w'))


def analyze_bbox_aoa():

    parser = imgcap_aoa_argument_parser()
    args = parser.parse_args()
    correctness = np.zeros((6, 10))
    for i, explanation_type in enumerate(['lrp_merge', 'attention_merge', 'GuidedBackpropagate_merge', 'gradient_merge', 'GuidedGradCam_merge', 'GradCam_merge']):
        count = 0
        correctness_sub = np.zeros(10)
        path_bbox = os.path.join(args.save_path, args.encoder, 'coco2017/evaluation/bbox/', explanation_type)
        print(path_bbox)
        files = glob.glob(os.path.join(path_bbox,'*.json'))
        print(len(files))
        for file in files:
            data = json.load(open(file, 'r'))
            # print(data)
            if data == {}:
                continue
            for category in data:
                item = data[category]
                count += 1
                for idx, th in enumerate([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
                    if item[str(th)] == 'nan':
                        item[str(th)] = 0
                    correctness_sub[idx] += float(item[str(th)])
        correctness_sub = correctness_sub / count
        correctness[i] = correctness_sub
        print(correctness[i])
        print(count)  #4649
    row_list = [['th']+[i/10 for i in range(10)]]
    for i, explanation_type in enumerate(['lrp', 'attention', 'GuidedBackpropagate', 'gradient', 'GuidedGradCam', 'GradCam']):
        row_list.append([explanation_type]+correctness[i].tolist())
    print(row_list)
    with open(os.path.join(args.save_path, args.encoder, 'coco2017/evaluation/bbox/', 'correctness.csv'), 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(row_list)


def analyze_TPFP_20(model_type):
    if model_type == 'gridTD':
        parser = imgcap_gridTD_argument_parser()
        args = parser.parse_args()
        explanation_type = ['lrp', 'GuidedBackpropagate', 'GuidedGradCam', 'GradCam', 'attention', 'beta']
    elif model_type == 'aoa':
        parser = imgcap_aoa_argument_parser()
        args = parser.parse_args()
        explanation_type = ['lrp', 'GuidedBackpropagate', 'gradient', 'GuidedGradCam', 'GradCam', 'attention']
    else:
        raise NotImplementedError('model_type in aoa, gridTD or adaptive')
    results_tp = {}
    results_fp = {}
    for i, explanation_type in enumerate(explanation_type):
        print(explanation_type)
        tp = {}
        fp = {}

        # print(explanation_type)
        tp_count = 0
        fp_count = 0
        quantile_list = [i / 100 for i in range(0, 51)]
        path_ablation = os.path.join(args.save_path, args.encoder, 'flickr30k', 'evaluation/tpfp/', explanation_type)
        if explanation_type in ['attention']:
            files_tp = glob.glob(os.path.join(args.save_path, args.encoder, 'flickr30k/evaluation/tpfp-beam1/lrp/',
                                           '*' + explanation_type + '_TP_statistics.json'))
            files_fp = glob.glob(os.path.join(args.save_path, args.encoder, 'flickr30k/evaluation/tpfp-beam1/lrp/',
                                              '*' + explanation_type + '_FP_statistics.json'))

            tp[explanation_type + 'mean'] = []
            tp[explanation_type + 'max'] = []
            fp[explanation_type + 'mean'] = []
            fp[explanation_type + 'max'] = []
            for i in range(len(quantile_list)):
                tp[explanation_type + 'quantile' + str(quantile_list[i])] = []
                fp[explanation_type + 'quantile' + str(quantile_list[i])] = []
            for file in files_tp:
                data = json.load(open(file, 'r'))
                if data == []:
                    continue
                statistics = {}
                for item in data:
                    word = item['word']
                    if word not in statistics.keys():
                        statistics[word] = {'mean': float('-inf'), 'max': float('-inf')}
                        for i in range(len(quantile_list)):
                            statistics[word]['quantile'+str(quantile_list[i])] = float('-inf')
                        tp_count+=1
                    if item['mean'] == "nan":
                        continue
                    else:
                        statistics[word]['mean'] = np.maximum(statistics[word]['mean'],float(item['mean']))
                    if item['max'] == 'nan':
                        continue
                    else:
                        statistics[word]['max'] = np.maximum(statistics[word]['max'],float(item['max']))
                    for i in range(len(quantile_list)):
                        statistics[word]['quantile' + str(quantile_list[i])] = np.maximum(statistics[word]['quantile'+str(quantile_list[i])], float(item['quantile'][i]))
                for key in statistics.keys():
                    tp[explanation_type+'mean'].append(statistics[key]['mean'])
                    tp[explanation_type+'max'].append(statistics[key]['max'])
                    for i in range(len(quantile_list)):
                        tp[explanation_type+'quantile'+str(quantile_list[i])].append(statistics[word]['quantile' + str(quantile_list[i])])
            for file in files_fp:
                data = json.load(open(file, 'r'))
                if data == []:
                    continue
                statistics = {}
                for item in data:
                    word = item['word']
                    if word not in statistics.keys():
                        statistics[word] = {'mean': float('inf'), 'max': float('inf')}
                        for i in range(len(quantile_list)):
                            statistics[word]['quantile'+str(quantile_list[i])] = float('inf')
                        fp_count += 1
                    if item['mean'] == "nan":
                        continue
                    else:
                        statistics[word]['mean'] = np.minimum(statistics[word]['mean'],float(item['mean']))
                    if item['max'] == 'nan':
                        continue
                    else:
                        statistics[word]['max'] = np.minimum(statistics[word]['max'],float(item['max']))
                    for i in range(len(quantile_list)):
                        statistics[word]['quantile' + str(quantile_list[i])] = np.minimum(statistics[word]['quantile'+str(quantile_list[i])], float(item['quantile'][i]))
                for key in statistics.keys():
                    fp[explanation_type+'mean'].append(statistics[key]['mean'])
                    fp[explanation_type+'max'].append(statistics[key]['max'])
                    for i in range(len(quantile_list)):
                        fp[explanation_type+'quantile' + str(quantile_list[i])].append(
                            statistics[word]['quantile' + str(quantile_list[i])])
            for key in tp.keys():
                results_tp[key]=tp[key]
                results_fp[key] = fp[key]
        elif explanation_type == 'beta':
            files_tp = glob.glob(os.path.join(args.save_path, args.encoder, 'flickr30k/evaluation/tpfp-beam1/lrp/',
                                           '*' + explanation_type + '_TP_statistics.json'))
            files_fp = glob.glob(os.path.join(args.save_path, args.encoder, 'flickr30k/evaluation/tpfp-beam1/lrp/',
                                              '*' + explanation_type + '_FP_statistics.json'))


            tp['1-beta'] = []
            fp['1-beta'] = []

            for file in files_tp:
                data = json.load(open(file, 'r'))
                if data == []:
                    continue
                statistics = {}
                for item in data:
                    word = item['word']
                    if word not in statistics.keys():
                        statistics[word] = {'1-beta': float('-inf')}
                        tp_count += 1
                    if item['1-beta'] == 'nan':
                        continue
                    else:
                        statistics[word]['1-beta'] = np.maximum(float(item['1-beta']),statistics[word]['1-beta'])
                for key in statistics.keys():
                    tp['1-beta'].append(statistics[key]['1-beta'])
            for file in files_fp:
                data = json.load(open(file, 'r'))
                if data == []:
                    continue
                statistics = {}
                for item in data:
                    word = item['word']
                    if word not in statistics.keys():
                        statistics[word] = {'1-beta': float('inf')}
                        fp_count += 1
                    if item['1-beta'] == 'nan':
                        continue
                    else:
                        statistics[word]['1-beta'] = np.minimum(float(item['1-beta']), statistics[word]['1-beta'])
                for key in statistics.keys():
                    fp['1-beta'].append(statistics[key]['1-beta'])
            for key in tp.keys():
                results_tp[key] = tp[key]
                results_fp[key] = fp[key]
        else:
            files_tp = glob.glob(os.path.join(path_ablation,'*' + explanation_type + '_TP_statistics.json'))
            files_fp = glob.glob(os.path.join(path_ablation,'*' + explanation_type + '_FP_statistics.json'))
            if explanation_type == 'lrp':
                files_tp = glob.glob(
                    os.path.join(args.save_path, args.encoder, 'flickr30k/evaluation/tpfp-beam1/lrp/',
                                 '*' + explanation_type + '_TP_statistics.json'))
                files_fp = glob.glob(
                    os.path.join(args.save_path, args.encoder, 'flickr30k/evaluation/tpfp-beam1/lrp/',
                                 '*' + explanation_type + '_FP_statistics.json'))

            for s in ['mean', 'mean_pos', 'max','mean_abs']:
                tp[explanation_type + s] = []
                fp[explanation_type + s] = []
            for i in range(len(quantile_list)):
                tp[explanation_type + 'quantile' + str(quantile_list[i])]=[]
                fp[explanation_type + 'quantile' + str(quantile_list[i])]=[]

            for file in files_tp:
                data = json.load(open(file, 'r'))
                if data == []:
                    continue
                statistics = {}
                for item in data:
                    word = item['word']
                    if word not in statistics.keys():
                        tp_count += 1
                        statistics[word] = {'mean': float('-inf'), 'mean_pos': float('-inf'), 'mean_abs': float('-inf'), 'max': float('-inf')}
                        for i in range(len(quantile_list)):
                            statistics[word]['quantile'+str(quantile_list[i])] = float('-inf')
                    if item['mean'] == 'nan':
                        continue
                    else:
                        statistics[word]['mean'] = np.maximum(statistics[word]['mean'],float(item['mean']))
                    if item['mean_pos'] == 'nan':
                        continue
                    else:
                        statistics[word]['mean_pos'] = np.maximum(statistics[word]['mean_pos'],float(item['mean_pos']))
                    if item['max'] == 'nan':
                        continue
                    else:
                        statistics[word]['max'] = np.maximum(statistics[word]['max'],float(item['max']))
                    if item['mean_abs'] == 'nan':
                        continue
                    else:
                        statistics[word]['mean_abs'] = np.maximum(statistics[word]['mean_abs'],float(item['mean_abs']))
                    for i in range(len(quantile_list)):
                        statistics[word]['quantile' + str(quantile_list[i])] = np.maximum(statistics[word]['quantile'+str(quantile_list[i])], float(item['quantile'][i]))
                for key in statistics.keys():
                    tp[explanation_type+'mean'].append(statistics[key]['mean'])
                    tp[explanation_type+'max'].append(statistics[key]['max'])
                    tp[explanation_type+'mean_pos'].append(statistics[key]['mean_pos'])
                    tp[explanation_type+'mean_abs'].append(statistics[key]['mean_abs'])
                    for i in range(len(quantile_list)):
                        tp[explanation_type+'quantile'+str(quantile_list[i])].append(statistics[word]['quantile' + str(quantile_list[i])])
            for file in files_fp:
                data = json.load(open(file, 'r'))
                if data == []:
                    continue
                statistics = {}
                for item in data:
                    word = item['word']
                    if word not in statistics.keys():
                        statistics[word] = {'mean': float('inf'), 'mean_pos': float('inf'), 'mean_abs': float('inf'), 'max': float('inf')}
                        for i in range(len(quantile_list)):
                            statistics[word]['quantile'+str(quantile_list[i])] = float('inf')
                    if item['mean'] == 'nan':
                        continue
                    else:
                        statistics[word]['mean'] = np.minimum(statistics[word]['mean'],float(item['mean']))
                    if item['mean_pos'] == 'nan':
                        continue
                    else:
                        statistics[word]['mean_pos'] = np.minimum(statistics[word]['mean_pos'],float(item['mean_pos']))
                    if item['max'] == 'nan':
                        continue
                    else:
                        statistics[word]['max'] = np.minimum(statistics[word]['max'],float(item['max']))
                    if item['mean_abs'] == 'nan':
                        continue
                    else:
                        statistics[word]['mean_abs'] = np.minimum(statistics[word]['mean_abs'],float(item['mean_abs']))
                    for i in range(len(quantile_list)):
                        statistics[word]['quantile' + str(quantile_list[i])] = np.minimum(statistics[word]['quantile'+str(quantile_list[i])], float(item['quantile'][i]))
                for key in statistics.keys():

                    fp[explanation_type+'mean'].append(statistics[key]['mean'])
                    fp[explanation_type+'max'].append(statistics[key]['max'])
                    fp[explanation_type+'mean_pos'].append(statistics[key]['mean_pos'])
                    fp[explanation_type+'mean_abs'].append(statistics[key]['mean_abs'])
                    for i in range(len(quantile_list)):
                        fp[explanation_type+'quantile'+str(quantile_list[i])].append(statistics[word]['quantile' + str(quantile_list[i])])
            for key in tp.keys():
                results_tp[key] = tp[key]
                results_fp[key] = fp[key]
            # results_tp[explanation_type+'mean'] = tp['mean']
            # results_fp[explanation_type+'mean'] = fp['mean']
            # results_tp[explanation_type+'max'] = tp['max']
            # results_fp[explanation_type+'max'] = fp['max']
            # results_tp[explanation_type+'mean_pos'] = tp['mean_pos']
            # results_fp[explanation_type+'mean_pos'] = fp['mean_pos']
            # results_tp[explanation_type+'mean_abs'] = tp['mean_abs']
            # results_fp[explanation_type+'mean_abs'] = fp['mean_abs']
    auc_score = {}
    for key in results_fp.keys():
        print(key, len(results_fp[key]), len(results_tp[key]))
        # print(results_fp[key])
        label_fp = [0] * len(results_fp[key])
        label_tp = [1] * len(results_tp[key])
        fpr, tpr, threshold = roc_curve(label_tp+label_fp, results_tp[key] + results_fp[key])
        roc_auc = auc(fpr, tpr)
        auc_score[key] = str(roc_auc)
        print(key, roc_auc)
    with open(os.path.join(args.save_path, args.encoder, 'flickr30k/evaluation/tpfp-beam1/', 'full_auc.json'),'w') as f:
        json.dump(auc_score, f)


def observe_frequent_words(predicted_file,th):
    f = yaml.safe_load(open(predicted_file, 'r'))
    vocab = {}
    f_list = []
    for key in f:
        sentence = f[key][0]
        words = sentence.split()
        for w in words:
            if w not in vocab.keys():
                vocab[w] = 0
            vocab[w] += 1
    sorted_vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
    # for key in sorted_vocab:
    #     if key not in STOP_WORDS:
    #         if sorted_vocab[key] > th:
    #             print(key, sorted_vocab[key])
    #             f_list.append(key)
    # print(f_list)
    for key in sorted_vocab:
        if key in flickr_frequent:
            print(key, sorted_vocab[key])


def count_hallucinate_words(predicted_file, gt_file, category_list):
    count_cat = {}
    tp = {}
    fp = {}
    tn = {}
    fn = {}
    precision = {}
    recall = {}
    f1_score = {}
    for cat in category_list:
        count_cat[cat] = 0
        tp[cat] = 0.
        fp[cat] = 0.
        tn[cat] = 0.
        fn[cat] = 0.
        precision[cat] = 0.
        recall[cat] = 0.
        f1_score[cat] = 0.
    predictions = yaml.safe_load(open(predicted_file, 'r'))
    gt_references = yaml.safe_load(open(gt_file, 'r'))
    acc_true = 0.
    total_true = 0.
    predict_true = 0.
    predict_list = []
    correct_predict_list = []
    for category in category_list:
        for key in predictions.keys():
            predicted_words = predictions[key][0].split(' ')
            gt_vocab = set()
            gt_sentences = gt_references[key]
            for sentence in gt_sentences:
                for word in sentence.split(' '):
                    gt_vocab.add(word)
            if category in predicted_words:
                predict_true += 1
                predict_list.append(key)
            if category in gt_vocab and category in predicted_words:
                correct_predict_list.append(key)
                acc_true += 1
                tp[category] += 1
            if category in gt_vocab and category not in predicted_words:
                fn[category] += 1
            if category not in gt_vocab and category in predicted_words:
                fp[category] += 1
            if category not in gt_vocab and category not in predicted_words:
                tn[category] += 1
            if category in gt_vocab:
                count_cat[category] += 1
                total_true += 1
    # print(count_cat)
    mpa = 0.
    mrc = 0.
    mf1 = 0.
    for key in count_cat.keys():
        if tp[key] + fp[key] > 0:
            precision[key] = tp[key] / (tp[key] + fp[key])
        if fn[key] + tp[key] > 0:
            recall[key] = tp[key] / (fn[key] + tp[key])
        if recall[key] + precision[key] > 0:
            f1_score[key] = 2 * precision[key] * recall[key] / (recall[key] + precision[key])
        mpa += precision[key]
        mrc += recall[key]
        mf1 += f1_score[key]
    # print(precision)
    mpa /= len(tp)
    mrc /= len(tp)
    mf1 /= len(tp)
    # print(mpa, mrc, mf1)
    return mpa, mrc, mf1




def ground_truth_work_frequency(dataset):

    assert dataset in {'coco2014', 'flickr8k', 'flickr30k', 'coco2017'}
    if dataset == 'flickr30k':
        f_list = flickr_frequent
        karpathy_json_path = './dataset/dataset_flickr30k.json'
    elif dataset == 'coco2017':
        f_list = coco_frequent
        karpathy_json_path = './dataset/dataset_coco2017.json'
    else:
        raise NotImplementedError
    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = {}

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            if img['split'] in ['train', 'restval']:
                for word in c['tokens']:
                    if word not in word_freq:
                        word_freq[word] = 0.
                    word_freq[word] += 1
    sorted_vocab = {k: v for k, v in sorted(word_freq.items(), key=lambda item: item[1])}
    for key in sorted_vocab:
        if key in f_list:
            print(key, sorted_vocab[key])
            f_list.append(key)
    print(f_list)



if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # generate_evaluation_files('gridTD', explainer_type='lrp', dataset='flickr30k', do_attention=True)
    # generate_evaluation_files('gridTD', explainer_type='GuidedGradCam', dataset='flickr30k', do_attention=False)
    # generate_evaluation_files('gridTD', explainer_type='GradCam', dataset='flickr30k', do_attention=False)
    # generate_evaluation_files('gridTD', explainer_type='GuidedGradient', dataset='flickr30k', do_attention=False)
    # generate_evaluation_files('gridTD', explainer_type='Gradient', dataset='flickr30k', do_attention=False)
    # generate_evaluation_files('gridTD', explainer_type='lrp', do_attention=True)
    # generate_evaluation_files('gridTD', explainer_type='GuidedGradCam', do_attention=False)
    # generate_evaluation_files('gridTD', explainer_type='GradCam', do_attention=False)
    # generate_evaluation_files('gridTD', explainer_type='GuidedGradient', do_attention=False)
    # generate_evaluation_files('gridTD', explainer_type='Gradient', do_attention=False)
    # generate_evaluation_files('gridTD', explainer_type='GuidedGradCam', do_attention=False)
    # generate_evaluation_files('gridTD', explainer_type='GuidedGradCamneg', do_attention=False)
    # generate_evaluation_files('gridTD', explainer_type='lrpneg', do_attention=False)
    # for i in range(8):
    #     generate_evaluation_files('aoa', explainer_type='lrp', head_idx=i, do_attention=False)
    # generate_evaluation_files('aoa', explainer_type='GuidedGradCam', do_attention=False)
    # generate_evaluation_files('aoa', explainer_type='GradCam', do_attention=False)
    # generate_evaluation_files('aoa', explainer_type='GuidedGradient', do_attention=False)
    # generate_evaluation_files('aoa', explainer_type='Gradient', do_attention=False)

    # analyze_ablation('gridTD')
    analyze_TPFP_20('gridTD')
    # analyze_bbox('gridTD')
    # analyze_ablation_aoa()
    # process_multihead_attention_bbox_aoa()
    # analyze_bbox_aoa()
    #
    '''observe the frequent words of the test set'''
    # predicted_file = './output/gridTD_BU_cideropt/vgg16/cocorobust/reference_cocorobust_split_test_beam_search_3.yaml'
    # predicted_file = './output/gridTD_BU_cideropt/vgg16/coco2017/predictions_coco2017_split_test_beam_search_3_epoch13.yaml'
    # predicted_file = './output/gridTD_BU_cideropt/vgg16/flickr30k/predictions_flickr30k_split_test_beam_search_3_epoch34.yaml'
    # observe_frequent_words(predicted_file,0)
    # ground_truth_work_frequency('flickr30k')

    '''calculate map'''

    # for model_type in ['gridTD', 'aoa']:
    #     for dataset_name in ['coco2017', 'flickr30k']:
    #         gt_reference = glob.glob(f'./output/{model_type}/vgg16/{dataset_name}/reference*')[0]
    #         for tune_type in [ 'baselinefinetune','lrpfinetune','baselineciderfinetune','lrpciderfinetune'  ]:
    #             prediction_paths = glob.glob(f'./output/{model_type}/vgg16/{dataset_name}/{tune_type}/predictions_*')[0]
    #             if dataset_name == 'coco2017':
    #                 f_list = coco_frequent
    #             elif dataset_name == 'flickr30k':
    #                 f_list = flickr_frequent
    #             print(model_type, dataset_name, tune_type)
    #             mpa, mrc, mf1 = count_hallucinate_words(prediction_paths, gt_reference, f_list)
    #             print(f'mpa:\t{mpa}\tmrc:\t{mrc}\tmf1:\t{mf1}')

    # for model_type in [ 'aoa_bu', 'gridTD_BU']:
    #     for dataset_name in ['coco2017','flickr30k']:
    #         gt_reference = glob.glob(f'./output/{model_type}/vgg16/{dataset_name}/reference*')[0]
    #         for tune_type in [ 'baselinefinetune','lrpfinetune']:
    #             prediction_paths = glob.glob(f'./output/{model_type}_{tune_type}/vgg16/{dataset_name}/predictions_*')[0]
    #             if dataset_name == 'coco2017':
    #                 f_list = coco_frequent
    #             elif dataset_name == 'flickr30k':
    #                 f_list = flickr_frequent
    #             print(model_type, dataset_name, tune_type)
    #             mpa, mrc, mf1 = count_hallucinate_words(prediction_paths, gt_reference, f_list)
    #             print(f'mpa:\t{mpa}\tmrc:\t{mrc}\tmf1:\t{mf1}')

