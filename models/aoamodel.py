import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import models.resnet as resnet
import models.vgg as vgg
import numpy as np
import math
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import gc
import os
from LRPtools import lrp_wrapper
from LRPtools import utils as LRPutil
import yaml
import skimage.transform
from nltk.corpus import stopwords
import copy
STOP_WORDS = list(set(stopwords.words('english')))
STOP_WORDS += ['<start>', '<end>', '<pad>', '<unk>']
BAD_ENDINGS = ['with','in','on','of','a','at','to','for','an','this','his','her','that','the']
class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return x+y


class Encoder(nn.Module):
    def __init__(self, encoder_type):
        super(Encoder, self).__init__()
        if encoder_type == 'resnet101':
            self.encoder = resnet.resnet101(pretrained=True)
            self.feat_dim = self.encoder.feat_dim
        elif encoder_type == 'renset50':
            self.encoder = resnet.resnet50(pretrained=True)
            self.feat_dim = self.encoder.feat_dim
        elif encoder_type == 'vgg16':
            base_model = vgg.vgg16(pretrained=True)
            self.encoder = base_model.features[0:-1]
            self.feat_dim = base_model.feat_dim
        else:
            raise NotImplementedError("the encoder_type does not exist, please add your encoder_type options")
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, img):
        encoded_image = self.encoder(img)
        average_feat = self.avgpool(encoded_image)
        return encoded_image, average_feat.squeeze()


class MultiHeadedDotAttention(nn.Module):
    def __init__(self, num_head, hidden_dim, dropout=0.3, project_k_v_flag=True, norm_q=True, aoa=True):
        super(MultiHeadedDotAttention, self).__init__()
        assert hidden_dim % num_head == 0
        self.d_k = hidden_dim // num_head  # the dim of each head
        self.num_head = num_head
        if norm_q:
            self.norm = nn.BatchNorm1d(hidden_dim, track_running_stats=True)
        else:
            self.norm = lambda x: x
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        if project_k_v_flag:
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.k_proj = lambda x: x
            self.v_proj = lambda x: x
        self.aoa = aoa
        if aoa:
            self.aoa_layer = nn.Sequential(nn.Linear(2*hidden_dim, 2 * hidden_dim), nn.GLU())
            self.add = Add()
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) #(bs, num_head, num_query, num_pixel)
        p_attn = torch.softmax(scores, dim=-1)  #(bs, num_head, num_query, num_pixel)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn  #(bs, num_head, num_query, dk) (bs, num_head, num_query, num_pixel)

    def forward(self, query, value, key):
        if query.dim() == 2:
            single_query = True
            query = query.unsqueeze(1) #(bs, num_query, hiddendim)
        else:
            single_query = False
        batch_size = query.size(0)
        query = query.transpose(1, 2)
        query = self.norm(query)
        query = query.transpose(1,2)
        query_ = self.q_proj(query).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2) #(bs, num_head, num_query, dk)
        key_ = self.k_proj(key).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2) #(bs, num_head, num_pixel, dk)
        value_ = self.v_proj(value).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2) #(bs, num_head, num_pixel, dk)
        x, alpha = self.attention(query_, key_, value_, dropout=nn.Dropout(0.1) if self.training else None)  #(bs, num_head, num_query, dk), (bs, num_head, num_query, num_pixel)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.d_k) #(bs, num_query, hiddendim)
        if self.aoa:
            x = self.aoa_layer(self.dropout(torch.cat([x, query], -1)))
            x = self.add(x, query)
        if single_query:
            x = x.squeeze(1)
            alpha = alpha.squeeze(1)
        return x, alpha  #(bs, num_head, num_query, num_pixel)


class AOAModel(nn.Module):
    '''
    '''
    EPS = LRPutil.EPSILON
    def __init__(self, embed_dim, hidden_dim, num_head, vocab_size, encoder_type):
        super(AOAModel, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.encoder_type = encoder_type
        self.vocab_size = vocab_size
        self.num_head = num_head
        if hidden_dim % num_head != 0:
            raise TypeError("the number of head should be dividable by the hidden dim")
        self.dropout = nn.Dropout(0.3)
        # the image encoder to generate image features (bs, C, H, W)
        self.img_encoder = Encoder(self.encoder_type)
        self.encoder_raw_dim = self.img_encoder.feat_dim
        print(f'==========Encoded image feature dim is {self.encoder_raw_dim}==========')
        self.img_projector = nn.Conv2d(self.encoder_raw_dim, self.hidden_dim, kernel_size=1,stride=1)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.LanguageLSTM = nn.LSTMCell(hidden_dim+embed_dim, hidden_dim)
        self.decoder_k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.decoder_v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.decoder_multihead_attention = MultiHeadedDotAttention(num_head=num_head, hidden_dim=hidden_dim, project_k_v_flag=False, norm_q=False, aoa=False)
        self.decoder_aoa_linear_gate = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.decoder_aoa_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()
        # self.refiner_batchnorm = nn.BatchNorm1d(hidden_dim, track_running_stats=True)

    def init_hidden_state(self, V):
        h = torch.zeros(V.shape[0], self.hidden_dim).cuda()
        c = torch.zeros(V.shape[0], self.hidden_dim).cuda()
        return h, c

    def predict_next_word(self,image_feature_proj, xt, states):
        '''
        :param image_feature_proj:  bs, num_pixel, hidden_dim
        :param xt: bs, hidden_dim + embedding dim
        :param states: (ht, ct, context) each with shape bs, hidden_dim
        :return:
        '''
        htm1, ctm1 = states  # (bs, hidden_dim, )
        ht, ct = self.LanguageLSTM(xt, (htm1, ctm1))
        key = self.decoder_k_proj(image_feature_proj) # batch_size, num_pixel, 2 * hiddendim this is the concatenated key and value
        value = self.decoder_v_proj(image_feature_proj)
        context, alpha_t = self.decoder_multihead_attention(ht, key, value) #(bs, hidden_dim) alpha: (bs, num_pixel)
        context_aoa_gate = self.decoder_aoa_linear_gate(ht)
        context_aoa_linear = self.decoder_aoa_linear(context)
        context_aoa =  torch.sigmoid(context_aoa_gate) * context_aoa_linear#(bs, hiddendim)
        predict_score_t = self.fc(self.dropout(context_aoa+ht))  # (bs, vocab_size)
        return predict_score_t, alpha_t, None, (ht, ct)

    def forward(self, images, encoded_captions, caption_lengths, ss_prob):
        """
        images: the encoded images from the encoder, of shape (batch_size, C, H, W)
        global_features: the global image features returned by the Encoder, of shape: (batch_size, hidden_dim)
        encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        """
        batch_size = images.size(0)
        image_features, avg_feature = self.img_encoder(images) # (bs, fea_dim, H, W), (bs, fea_dim)
        # print(image_features.size(), avg_feature.size())
        image_feature_proj = self.relu(self.img_projector(image_features)) # (bs, hiddendim, H, W)
        # print(image_feature_proj.size())
        image_feature_proj = image_feature_proj.contiguous()
        image_feature_proj = image_feature_proj.view(batch_size, self.hidden_dim, -1) # (bs, hidden_dim, num_pixel)
        # print(image_feature_proj.size(-1))
        image_feature_proj = image_feature_proj.transpose(1,2) #(bs, num_pixel, hidden_dim)
        global_img_feature = torch.mean(image_feature_proj, dim=1) #(bs, hidden_dim)
        h, c = self.init_hidden_state(image_feature_proj)
        state = (h, c)
        max_length = max(caption_lengths)-1
        predictions = torch.zeros(batch_size, max_length, self.vocab_size).cuda()
        if ss_prob is None:
            ss_flag = False
        else:
            random_num = np.random.uniform(0.0, 1.0, size=(batch_size,))
            ss_mask = random_num < ss_prob
            ss_mask = torch.from_numpy(ss_mask).long().cuda()
            if ss_mask.sum() > 0:
                ss_flag = True
            else:
                ss_flag = False
        for t in range(max_length):
            if t>2 and ss_flag:
                it = last_label*ss_mask + encoded_captions[:, t] * (1-ss_mask)
                word_embedding = self.embedding(it) # (batch_size, embed_dim)
            else:
                # print(t,max_length,encoded_captions)
                word_embedding = self.embedding(encoded_captions[:,t])
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            xt = torch.cat((word_embedding, global_img_feature), dim=-1)   # (batch_size, hidden_dim + embed_dim)
            predict_score_t, alpha_t, beta_t, state = self.predict_next_word(image_feature_proj, xt, state)
            predictions[:, t,:] = predict_score_t
            last_scores = torch.log_softmax(predict_score_t,-1)
            # print(last_scores.size())
            last_label = torch.argmax(last_scores, -1)  #(batch_size, )
            # print(last_label.size())
        return predictions, None, None, last_scores, max_length

    def sample(self, images, word_map, caption_lengths, opt={}):

        batch_size = images.size(0)
        sample_method = opt.get('sample_method', 'greedy')
        temperature = opt.get('temperature', 1.0)
        max_length = max(caption_lengths) - 1
        image_features, avg_feature = self.img_encoder(images)  # (bs, fea_dim, H, W), (bs, fea_dim)
        image_feature_proj = self.relu(self.img_projector(image_features))  # (bs, hiddendim, H, W)
        image_feature_proj = image_feature_proj.contiguous()
        image_feature_proj = image_feature_proj.view(batch_size, self.hidden_dim, -1)  # (bs, hidden_dim, num_pixel)
        image_feature_proj = image_feature_proj.transpose(1, 2)  # (bs, num_pixel, hidden_dim)
        global_img_feature = torch.mean(image_feature_proj, dim=1)  # (bs, hidden_dim)
        h, c = self.init_hidden_state(image_feature_proj)
        state = (h, c)
        seq = torch.zeros(batch_size,max_length).long().cuda()
        seq_logprobs = torch.zeros(batch_size, max_length).cuda()
        for t in range(max_length):
            if t == 0:
                it = torch.ones(batch_size).long().cuda() * word_map['<start>']
            word_embedding = self.embedding(it)
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            xt = torch.cat((word_embedding,  global_img_feature), dim=-1)  # (batch_size, 2*embed_dim)
            predict_score_t, alpha_t, beta_t, state = self.predict_next_word(image_feature_proj, xt, state)
            predict_score_t = torch.log_softmax(predict_score_t,dim=-1)
            it, sampleLpgprobs = self.sample_next_word(predict_score_t, sample_method, temperature)
            # sample the next word
            if t == 0:
                current_finished = it == word_map['<end>']
                unfinished = ~current_finished
            else:
                current_finished = it == word_map['<end>']
                unfinished = unfinished * ~current_finished
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            seq_logprobs[:, t] = sampleLpgprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break
        return seq, seq_logprobs, max_length

    def sample_next_word(self, logprobs, sample_method, temperature):
        if sample_method == 'greedy':
            sampleLogprobs, it = torch.max(logprobs.detach(), 1)
            it = it.view(-1).long()
        elif sample_method == 'gumbel': # gumbel softmax
            def sample_gumbel(shape, eps=1e-20):
                U = torch.rand(shape).cuda()
                return -torch.log(-torch.log(U + eps) + eps)
            def gumbel_softmax_sample(logits, temperature):
                y = logits + sample_gumbel(logits.size())
                return torch.log_softmax(y / temperature, dim=-1)
            _logprobs = gumbel_softmax_sample(logprobs, temperature)
            _, it = torch.max(_logprobs.data, 1)
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1)) # gather the logprobs at sampled positions
        else:
            logprobs = logprobs / temperature
            if sample_method.startswith == 'top': # topk sampling
                top_num = float(sample_method[3:])
                if 0 < top_num < 1:
                    # nucleus sampling from # The Curious Case of Neural Text Degeneration
                    probs = torch.softmax(logprobs, dim=1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
                    _cumsum = sorted_probs.cumsum(1)
                    mask = _cumsum < top_num
                    mask = torch.cat([torch.ones_like(mask[:,:1]), mask[:,:-1]], 1)
                    sorted_probs = sorted_probs * mask.float()
                    sorted_probs = sorted_probs / sorted_probs.sum(1, keepdim=True)
                    logprobs.scatter_(1, sorted_indices, sorted_probs.log())
                else:
                    the_k = int(top_num)
                    tmp = torch.empty_like(logprobs).fill_(float('-inf'))
                    topk, indices = torch.topk(logprobs, the_k, dim=1)
                    tmp = tmp.scatter(1, indices, topk)
                    logprobs = tmp
            it = torch.distributions.Categorical(logits=logprobs.detach()).sample()  #(batch_size,)
            # print(it.size())
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1)) # gather the logprobs at sampled positions (bs, 1)
            # print(sampleLogprobs.size())
        return it, sampleLogprobs

    def remove_bad_endings(self, sentences):
        new_sentences = []
        for sentence in sentences:
            words = sentence.split(' ')
            while words[-1] in BAD_ENDINGS:
                words = words[:-1]
            new_sentence = ' '.join(words)
            new_sentences.append(new_sentence)
        return new_sentences

    def diverse_beam_search(self,imgs, beam_size,word_map, max_cap_length=50, diversity_prob=0.5): # only support batch_size 1
        '''
        This function only suits for batch_size 1
        :param imgs:
        :param model:
        :param word_map:
        :param beam_size:
        :param max_cap_length:
        :return:
        '''
        self.eval()
        vocab_size = len(word_map)
        num_group = beam_size
        batch_size = imgs.size(0)
        assert batch_size == 1
        rev_word_map = {v: k for k, v in word_map.items()}
        with torch.no_grad():
            complete_seqs = [[] for g in range(num_group)]
            complete_seqs_scores = [[] for g in range(num_group)]
            k_prev_words = [torch.LongTensor([[word_map['<start>']]] *beam_size).cuda() for g in range(num_group)] # (beam_size,)
            top_k_scores = [torch.zeros(beam_size, 1).cuda() for g in range(num_group)] # (beam_size, 1)
            seqs = [torch.LongTensor([[word_map['<start>']]] *beam_size).cuda() for g in range(num_group)]   # (unfinished_num, )
            image_features, avg_feature = self.img_encoder(imgs) #
            image_feature_proj = self.relu(self.img_projector(image_features)) # batch_size, hidden_dim, H, W
            image_feature_proj = image_feature_proj.contiguous()
            image_feature_proj = image_feature_proj.view(batch_size, self.hidden_dim, -1)  # (bs, hidden_dim, num_pixel)
            image_feature_proj = image_feature_proj.transpose(1, 2)  # (bs, num_pixel, hidden_dim)
            global_img_feature = torch.mean(image_feature_proj, dim=1)  # (bs, hidden_dim)

            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)   # batch_size, hidden_dim
            image_feature_proj = [image_feature_proj.expand(beam_size, *image_feature_proj.size()[1:]) for g in range(num_group)]  # batch_size, H*W, hidden_dim
            global_img_feature = [global_img_feature.expand(beam_size, global_img_feature.size(-1)) for g in range(num_group)] #  beam_size, hidden_dim,
            h, c = self.init_hidden_state(image_feature_proj[0])
            init_state = (h, c)
            state = [init_state for g in range(num_group)]  #(ht, ct)
            unfinished_num = [beam_size for g in range(num_group)]
            for step in range(max_cap_length):
                previous_idx = []
                for g in range(num_group):
                    if unfinished_num[g] == 0:
                        continue
                    word_embedding = self.embedding(k_prev_words[g]).squeeze(1)  # unfinished_num, embedding_dim
                    xt = torch.cat((word_embedding,  global_img_feature[g]), dim=-1)  # (batch_size, embed_dim + hidden_dim)
                    # print(image_feature_proj[g].size(), xt.size(), state[g])
                    predict_score_t, alpha_t, beta_t, state[g] = self.predict_next_word(image_feature_proj[g], xt, state[g])
                    predict_score_t = torch.log_softmax(predict_score_t, dim=-1)  # (unfinished_num, vocab_size)
                    for i, v in enumerate(previous_idx):
                        predict_score_t[:,int(v)] = predict_score_t[:, int(v)] - diversity_prob
                    top_k_scores_exp = top_k_scores[g].expand((unfinished_num[g], vocab_size))
                    scores = top_k_scores_exp + predict_score_t
                    if step == 0:
                        top_k_scores[g], top_words = scores[0].topk(beam_size, -1, True, True)  # (unfinished_num, beam_size)
                    else:
                        top_k_scores[g], top_words = scores.view(-1).topk(unfinished_num[g], -1, True, True)  # (unfinished_num, beam_size)
                    beam_idx = top_words / vocab_size  # (unfinished_num, )
                    next_word_idx = top_words % vocab_size  # (unfinished_num, )
                    # print('next_word',next_word_idx)
                    seqs[g] = torch.cat([seqs[g][beam_idx], next_word_idx.unsqueeze(1)], dim=1)
                    incomplete_inds = [ind for ind, next_word in enumerate(next_word_idx) if next_word != word_map['<end>']]
                    complete_inds = list(set(range(len(next_word_idx))) - set(incomplete_inds))
                    # Set aside complete sequences
                    if len(complete_inds) > 0:
                        complete_seqs[g].extend(seqs[g][complete_inds].tolist())
                        complete_seqs_scores[g].extend(top_k_scores[g][complete_inds])
                    unfinished_num[g] = unfinished_num[g] - len(complete_inds)  # reduce beam length accordingly
                    if unfinished_num[g] == 0:
                        break
                    # updata sequences
                    seqs[g] = seqs[g][incomplete_inds]
                    #  update state
                    new_state = []
                    for s_idx in range(len(state[g])):
                        new_state.append(state[g][s_idx][beam_idx[incomplete_inds]])
                    state[g] = tuple(new_state)
                    image_feature_proj[g] = image_feature_proj[g][beam_idx[incomplete_inds]]
                    global_img_feature[g] = global_img_feature[g][beam_idx[incomplete_inds]]
                    top_k_scores[g] = top_k_scores[g][incomplete_inds].unsqueeze(1)
                    if g < 2:
                        for i, v in enumerate(k_prev_words[g]):
                            if v.item() not in previous_idx:
                                previous_idx.append(v.item())
                    k_prev_words[g] = next_word_idx[incomplete_inds].unsqueeze(1)

            return_sentences = []
            for g in range(num_group):
                if len(complete_seqs[g]) > 0:
                    i = complete_seqs_scores[g].index(max(complete_seqs_scores[g]))
                    seq = complete_seqs[g][i]
                else:
                    seq = seqs[0][0][:20]
                    seq = [seq[i].item() for i in range(len(seq))]
                sen_idx = [w for w in seq if
                           w not in {word_map['<start>'], word_map['<end>'], word_map['<unk>'], word_map['<pad>']}]
                sentence = ' '.join([rev_word_map[sen_idx[i]] for i in range(len(sen_idx))])
                print(sentence)
                return_sentences.append(sentence)
            return_sentences = self.remove_bad_endings(return_sentences)
            return return_sentences

    def beam_search(self,imgs,  word_map, beam_size=3,max_cap_length=30):
        '''
        This function only suits for batch_size 1
        :param imgs:
        :param model:
        :param word_map:
        :param beam_size:
        :param max_cap_length:
        :return:
        '''
        self.eval()
        assert imgs.size(0) == 1
        batch_size = imgs.size(0)
        rev_word_map = {v: k for k, v in word_map.items()}
        vocab_size = len(word_map)
        complete_seqs =[]
        complete_seqs_scores=[]
        with torch.no_grad():
            k_prev_words = torch.LongTensor([[word_map['<start>']]] *beam_size).cuda() # (beam_size,)
            top_k_scores = torch.zeros(beam_size, 1).cuda() # (beam_size, 1)
            seqs = torch.LongTensor([[word_map['<start>']]] *beam_size).cuda()  # (unfinished_num, )
            image_features, avg_feature = self.img_encoder(imgs) #
            image_feature_proj = self.relu(self.img_projector(image_features)) # batch_size, hidden_dim, H, W
            image_feature_proj = image_feature_proj.contiguous()
            image_feature_proj = image_feature_proj.view(batch_size, self.hidden_dim, -1)  # (bs, hidden_dim, num_pixel)
            image_feature_proj = image_feature_proj.transpose(1, 2)  # (bs, num_pixel, hidden_dim)
            global_img_feature = torch.mean(image_feature_proj, dim=1)  # (bs, hidden_dim)
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)   # batch_size, hidden_dim
            # print(global_img_feature.size())
            image_feature_proj = image_feature_proj.expand(beam_size, *image_feature_proj.size()[1:])  # batch_size, H*W, hidden_dim
            global_img_feature = global_img_feature.expand(beam_size, global_img_feature.size(-1)) #  beam_size, hidden_dim,
            h, c = self.init_hidden_state(image_feature_proj)
            state = (h, c)
            unfinished_num = beam_size
            for step in range(max_cap_length):
                word_embedding = self.embedding(k_prev_words).squeeze(1) # unfinished_num, embedding_dim
                xt = torch.cat((word_embedding, global_img_feature), dim=-1) # (batch_size, 2*embed_dim + hidden_dim)
                predict_score_t, alpha_t, beta_t, state = self.predict_next_word(image_feature_proj, xt, state)
                predict_score_t = torch.log_softmax(predict_score_t,dim=-1) #(unfinished_num, vocab_size)
                top_k_scores_exp = top_k_scores.expand((unfinished_num, vocab_size))
                scores = top_k_scores_exp + predict_score_t
                if step == 0:
                    top_k_scores, top_words = scores[0].topk(beam_size, -1, True, True)  # (unfinished_num, beam_size)
                else:
                    top_k_scores, top_words = scores.view(-1).topk(unfinished_num, -1, True, True) # (unfinished_num, beam_size)
                beam_idx = top_words // vocab_size  # (unfinished_num, )
                next_word_idx = top_words % vocab_size  # (unfinished_num, )
                # print('next_word',next_word_idx)
                seqs = torch.cat([seqs[beam_idx], next_word_idx.unsqueeze(1)], dim=1)
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_idx) if next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_idx))) - set(incomplete_inds))
                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                unfinished_num = unfinished_num - len(complete_inds)  # reduce beam length accordingly
                if unfinished_num == 0:
                    break
                # updata sequences
                seqs = seqs[incomplete_inds]
                #  update state
                new_state = []
                for s_idx in range(len(state)):
                    new_state.append(state[s_idx][beam_idx[incomplete_inds]])
                state = tuple(new_state)
                image_feature_proj = image_feature_proj[beam_idx[incomplete_inds]]
                global_img_feature = global_img_feature[beam_idx[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_idx[incomplete_inds].unsqueeze(1)
                # Break if things have been going on too long
            if len(complete_seqs) > 0:
                i = complete_seqs_scores.index(max(complete_seqs_scores))
                seq = complete_seqs[i]
            else:
                seq = seqs[0][:20]
                seq = [seq[i].item() for i in range(len(seq))]
            sen_idx = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<unk>'], word_map['<pad>']}]
            sentence = [' '.join([rev_word_map[sen_idx[i]] for i in range(len(sen_idx))])]
            sentence = self.remove_bad_endings(sentence)
            return sentence, sen_idx

    def greedy_search(self,imgs,  word_map, max_cap_length=20):
        self.eval()
        batch_size = imgs.size(0)
        rev_word_map = {v: k for k, v in word_map.items()}
        complete_seq =[]
        with torch.no_grad():
            k_prev_words = torch.zeros(batch_size, max_cap_length).long().cuda() # (batch_size, caption_length)
            k_prev_words[:, 0] = word_map['<start>'] # the first word is '<start>'
            seqs_temp = [[word_map['<start>']] for _ in range(batch_size)]
            image_features, avg_feature = self.img_encoder(imgs) #
            image_feature_proj = self.relu(self.img_projector(image_features)) # batch_size, hidden_dim, H, W
            image_feature_proj = image_feature_proj.contiguous()
            image_feature_proj = image_feature_proj.view(batch_size, self.hidden_dim, -1)  # (bs, hidden_dim, num_pixel)
            image_feature_proj = image_feature_proj.transpose(1, 2)  # (bs, num_pixel, hidden_dim)
            global_img_feature = torch.mean(image_feature_proj, dim=1)  # (bs, hidden_dim)
            h, c = self.init_hidden_state(image_feature_proj)
            state = (h, c)
            for step in range(max_cap_length-1):
                word_embedding = self.embedding(k_prev_words[:, step]) # batch_size, embedding_dim
                if global_img_feature.dim() == 1:
                    global_img_feature = global_img_feature.unsqueeze(0)
                xt = torch.cat((word_embedding,  global_img_feature), dim=-1) # (batch_size, embed_dim + hidden_dim)
                predict_score_t, alpha_t, beta_t, state = self.predict_next_word(image_feature_proj, xt, state)
                predict_score_t = torch.log_softmax(predict_score_t,dim=-1) #(batch_size, vocab_size)
                top_scores, top_words = predict_score_t.topk(1, -1, True, True)
                if step == 0:
                    finished = top_words == word_map['<end>']
                    unfinished = ~finished
                else:
                    current_finished = top_words == word_map['<end>']
                    unfinished = unfinished * ~current_finished
                top_words = top_words * unfinished.type_as(top_words)
                k_prev_words[:, step+1] = top_words[:,0]
                for bs in range(batch_size):
                    seqs_temp[bs].append(int(top_words[bs][0].cpu().numpy()))
            for bs in range(batch_size):
                sen = seqs_temp[bs]
                sen_idx = [w for w in sen if w not in {word_map['<start>'], word_map['<end>'], word_map['<unk>'],word_map['<pad>']}]
                # print(sen_idx)
                sentence = ' '.join([rev_word_map[sen_idx[i]] for i in range(len(sen_idx))])
                # print(sentence)
                complete_seq.append(sentence)
            complete_seq = self.remove_bad_endings(complete_seq)
            return complete_seq, sen_idx

    def lrp_linear_eps(self, r_out, forward_input, forward_output, weight):
        '''

        :param r_out:  relevance of the output (out_feature,)
        :param forward_input: the output tensor (in_feature, )
        :param forward_output: the input tensor (out_feature, )
        :param weight:  weight tensor shape (out_feature, in_feature)
        :return: r_in (in_feature,)
        '''
        assert r_out.dim() == 1
        assert forward_input.dim() == 1
        assert weight.dim() == 2
        attribution = weight * forward_input #(out_feature, in_feature)
        if type(forward_output) == bool:
            forward_output = torch.matmul(forward_input, weight.transpose(0,1))
            # print('matml', forward_output.size())
        forward_output_eps = self.EPS * forward_output.sign() + forward_output  # Z.sign() returns -1 or 0 or 1

        forward_output_eps.masked_fill_(forward_output_eps == 0, self.EPS)  #(out_feature,)
        # print(forward_output_eps.size())
        attribution_norm = attribution.transpose(0,1) / forward_output_eps #(in_feature, out_feature)
        # print(attribution_norm.size())
        relevance_input = torch.sum(attribution_norm * r_out, dim=-1) #(in_feature,)
        assert relevance_input.size() == forward_input.size()
        torch.cuda.empty_cache()
        return relevance_input

    def lrp_mha(self, alpha, value, r_context, context):
        '''

        :param alpha:  shape is num_head, num_pixel
        :param value: shape is  num_pixel hiddendim
        :param r_context: shape is  1, hiddendim
        :param context shape is 1, hiddendim
        :return:
        '''
        num_head = alpha.size(0)
        num_pixel = alpha.size(1)
        num_query = r_context.size(0)  #(should be 1 if we use ht ad the query)
        '''using mean'''
        # r_value = torch.zeros_like(value).cuda()
        # attention_weight = alpha.mean(0)
        # for i in range(num_pixel):
        #     r_value[i] = self.lrp_linear_eps(r_out=r_context.squeeze(),
        #                                      forward_input=value[i]*attention_weight[i],
        #                                      forward_output=context.squeeze(),
        #                                      weight=torch.eye(self.model.hidden_dim).cuda())
        '''spread single head'''
        d_k = self.model.hidden_dim//num_head #(should be 64 if using vgg and 8 heads)
        r_context = r_context.clone().contiguous().view(num_query, num_head, d_k).transpose(0, 1) #(num_head, num_query, d_k)
        context = context.clone().contiguous().view(num_query, num_head, d_k).transpose(0,1)  #(num_head, num_query, d_k)
        value = value.clone().contiguous().view(num_pixel, num_head, d_k).transpose(0, 1) #(num_head, num_pixel, d_k)
        r_value = value.clone()
        for h in range(num_head):
            for i in range(num_pixel):
                r = self.lrp_linear_eps(r_out=r_context[h,0],
                                                forward_input=value[h,i]*alpha[h,i],
                                                forward_output=context[h,0],
                                                weight=torch.eye(d_k).cuda())
                r_value[h, i] = r
        # print(r_value[:, 0].squeeze())
        r_value = r_value.transpose(0, 1).contiguous().view(num_pixel, self.model.hidden_dim)
        # print(r_value[0])
        return r_value

    def get_lrp_weight_step(self, predictions_t, rev_word_map, ht_, context_aoa):
        batch_size, vocab_size = predictions_t.size()
        with torch.no_grad():
            weight_of_context_aoa = torch.zeros(batch_size, self.hidden_dim).cuda()
            weight_of_ht = torch.zeros(batch_size, self.hidden_dim).cuda()
            for b in range(batch_size):
                predicted_labels = torch.argmax(predictions_t[b], dim=-1)  # (the predicted label of image b)  (max_length)
                word_t = predicted_labels.item()
                if rev_word_map[word_t] in STOP_WORDS + ['<start>','<end>','<pad>','<unk>']:
                    continue
                else:
                    word_relevance = torch.zeros(self.vocab_size).cuda()
                    word_relevance[word_t] = predictions_t[b][word_t]
                    r_h2t_context_aoa = self.lrp_linear_eps(r_out=word_relevance,
                                                            forward_input=ht_[b] + context_aoa[b],
                                                            forward_output=predictions_t[b],
                                                            weight=self.fc.weight)
                    r_h2t = self.lrp_linear_eps(r_out=r_h2t_context_aoa,
                                                forward_input=ht_[b],
                                                forward_output=ht_[b] + context_aoa[b],
                                                weight=torch.eye(self.hidden_dim).cuda())
                    weight_of_ht[b] = r_h2t
                    r_context_aoa = self.lrp_linear_eps(r_out=r_h2t_context_aoa,
                                                        forward_input=context_aoa[b],
                                                        forward_output=ht_[b] + context_aoa[b],
                                                        weight=torch.eye(self.hidden_dim).cuda())
                    weight_of_context_aoa[b] = r_context_aoa
            weight_of_context_aoa = LRPutil.normalize_relevance(weight_of_context_aoa, dim=-1)
            weight_of_ht = LRPutil.normalize_relevance(weight_of_ht, dim=-1)
            return weight_of_context_aoa, weight_of_ht

    def forwardlrp_context(self, images, encoded_captions, caption_lengths, rev_word_map):
        def lstm_forward(x, h, c, wi, wh, bi, bh):
            z = torch.matmul(x, wi.transpose(0, 1))  # (batch_size, 4*hidden_size)
            z = z + torch.matmul(h, wh.transpose(0, 1))  # (batch_size, 4*hidden_size)
            z = z + bi + bh
            z0, z1, z2, z3 = z.chunk(4, dim=1)
            i = torch.sigmoid(z0)
            f = torch.sigmoid(z1)
            c = f * c + i * torch.tanh(z2)
            o = torch.sigmoid(z3)
            h = o * torch.tanh(c)
            c = c
            return h, c, z2, i, f  # (batch_size, 512)
        batch_size = images.size(0)
        image_features, avg_feature = self.img_encoder(images)  # (bs, fea_dim, H, W), (bs, fea_dim)
        # print(image_features.size(), avg_feature.size())
        image_feature_proj_before_act = self.img_projector(image_features)
        image_feature_proj = self.relu(image_feature_proj_before_act)  # (bs, hiddendim, H, W)
        # print(image_feature_proj.size())
        image_feature_proj = image_feature_proj.contiguous()
        image_feature_proj = image_feature_proj.view(batch_size, self.hidden_dim, -1)  # (bs, hidden_dim, num_pixel)
        # num_pixels = image_feature_proj.size(-1)
        image_feature_proj = image_feature_proj.transpose(1, 2)  # (bs, num_pixel, hidden_dim)
        h, c = self.init_hidden_state(image_feature_proj)
        state = (h, c)
        global_img_feature = torch.mean(image_feature_proj, dim=1)  # (bs, hidden_dim)
        key = self.decoder_k_proj(image_feature_proj)
        value = self.decoder_v_proj(image_feature_proj)
        max_length = max(caption_lengths) - 1
        predictions = torch.zeros(batch_size, max_length, self.vocab_size).cuda()
        weighted_predictions = torch.zeros(batch_size, max_length, self.vocab_size).cuda()
        for t in range(max_length):
            word_embedding = self.embedding(encoded_captions[:, t])
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            xt_ = torch.cat((word_embedding, global_img_feature), dim=-1)  # (batch_size, hidden_dim + embed_dim)
            h_, c_, g_, i_act_, f_act_ = lstm_forward(xt_, state[0], state[1], self.LanguageLSTM.weight_ih,
                                                      self.LanguageLSTM.weight_hh,
                                                      self.LanguageLSTM.bias_ih, self.LanguageLSTM.bias_hh)
            context_, alpha_t_ = self.decoder_multihead_attention(h_, key, value)  # (bs, hidden_dim) alpha: (bs, num_pixel)
            context_aoa_gate_ = self.decoder_aoa_linear_gate(h_)
            context_aoa_linear_ = self.decoder_aoa_linear(context_)
            context_aoa_ = torch.sigmoid(context_aoa_gate_) * context_aoa_linear_  # (bs, hiddendim)
            predict_score_t = self.fc(self.dropout(context_aoa_ + h_))  # (bs, vocab_size)
            state = (h_, c_)
            predictions[:, t, :] = predict_score_t
            weight_context_aoa, weight_ht = self.get_lrp_weight_step(predict_score_t, rev_word_map, h_, context_aoa_)
            weighted_prediction_t = self.fc(self.dropout(weight_context_aoa*context_aoa_ + h_*weight_ht))
            weighted_predictions[:, t,:] = weighted_prediction_t
        return predictions, weighted_predictions, max_length

    def sample_lrp(self, images, rev_word_map, word_map, caption_lengths, opt={}):
        def lstm_forward(x, h, c, wi, wh, bi, bh):
            z = torch.matmul(x, wi.transpose(0, 1))  # (batch_size, 4*hidden_size)
            z = z + torch.matmul(h, wh.transpose(0, 1))  # (batch_size, 4*hidden_size)
            z = z + bi + bh
            z0, z1, z2, z3 = z.chunk(4, dim=1)
            i = torch.sigmoid(z0)
            f = torch.sigmoid(z1)
            c = f * c + i * torch.tanh(z2)
            o = torch.sigmoid(z3)
            h = o * torch.tanh(c)
            c = c
            return h, c, z2, i, f  # (batch_size, 512)
        batch_size = images.size(0)
        sample_method = opt.get('sample_method', 'greedy')
        temperature = opt.get('temperature', 1.0)
        max_length = max(caption_lengths) - 1
        image_features, avg_feature = self.img_encoder(images)  # (bs, fea_dim, H, W), (bs, fea_dim)
        image_feature_proj = self.relu(self.img_projector(image_features))  # (bs, hiddendim, H, W)
        image_feature_proj = image_feature_proj.contiguous()
        image_feature_proj = image_feature_proj.view(batch_size, self.hidden_dim, -1)  # (bs, hidden_dim, num_pixel)
        image_feature_proj = image_feature_proj.transpose(1, 2)  # (bs, num_pixel, hidden_dim)
        global_img_feature = torch.mean(image_feature_proj, dim=1)  # (bs, hidden_dim)
        key = self.decoder_k_proj(image_feature_proj)
        value = self.decoder_v_proj(image_feature_proj)
        h, c = self.init_hidden_state(image_feature_proj)
        state = (h, c)
        seq = torch.zeros(batch_size,max_length).long().cuda()
        seq_logprobs = torch.zeros(batch_size, max_length).cuda()
        for t in range(max_length):
            if t == 0:
                it = torch.ones(batch_size).long().cuda() * word_map['<start>']
            word_embedding = self.embedding(it)
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            xt_ = torch.cat((word_embedding, global_img_feature), dim=-1)  # (batch_size, hidden_dim + embed_dim)
            h_, c_, g_, i_act_, f_act_ = lstm_forward(xt_, state[0], state[1], self.LanguageLSTM.weight_ih,
                                                      self.LanguageLSTM.weight_hh,
                                                      self.LanguageLSTM.bias_ih, self.LanguageLSTM.bias_hh)
            context_, alpha_t_ = self.decoder_multihead_attention(h_, key,
                                                                  value)  # (bs, hidden_dim) alpha: (bs, num_pixel)
            context_aoa_gate_ = self.decoder_aoa_linear_gate(h_)
            context_aoa_linear_ = self.decoder_aoa_linear(context_)
            context_aoa_ = torch.sigmoid(context_aoa_gate_) * context_aoa_linear_  # (bs, hiddendim)
            predict_score_t = self.fc(self.dropout(context_aoa_ + h_))  # (bs, vocab_size)
            state = (h_, c_)
            predict_score_t = torch.log_softmax(predict_score_t,dim=-1)

            weight_context_aoa, weight_ht = self.get_lrp_weight_step(predict_score_t, rev_word_map, h_, context_aoa_)
            weight_prediction_t = self.fc(context_aoa_ * weight_context_aoa + weight_ht * h_)
            predict_score_t = torch.log_softmax(weight_prediction_t, dim=-1)

            it, sampleLpgprobs = self.sample_next_word(predict_score_t, sample_method, temperature)
            # sample the next word
            if t == 0:
                current_finished = it == word_map['<end>']
                unfinished = ~current_finished
            else:
                current_finished = it == word_map['<end>']
                unfinished = unfinished * ~current_finished
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            seq_logprobs[:, t] = sampleLpgprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break
        return seq, seq_logprobs, max_length

'''explainers'''
class ExplainAOAAttention(object):
    EPS = LRPutil.EPSILON
    EX_TYPE = 'lrp'
    def __init__(self, args, word_map, model=None):
        super(ExplainAOAAttention, self).__init__()
        self.args = args
        self.word_map = word_map
        self.rev_word_map = {v: k for k, v in word_map.items()}
        self.vocab_size = len(word_map)
        self.num_head = args.num_head
        if model is not None:
            self.model = model
        else:
            self.model = AOAModel(args.embed_dim, args.hidden_dim, args.num_head, len(word_map), args.encoder)
            checkpoint = torch.load(args.weight)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.cuda()
        self.model.eval()
        self.model.decoder_multihead_attention.eval()

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.img_transform = transforms.Compose([transforms.Resize(size=(args.height, args.width)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=self.mean, std=self.std)])

        self.language_weight_i = self.model.LanguageLSTM.weight_ih #(4*hidden_size, hiddendim+embed_dim)
        self.language_weight_h = self.model.LanguageLSTM.weight_hh #(4*hidden_size, hiddendim)
        self.language_bias_i = self.model.LanguageLSTM.bias_ih #(4*hidden_size,)
        self.language_bias_h = self.model.LanguageLSTM.bias_hh #(4*hidden_size,)

        self.output_weight = self.model.fc.weight  #(vocab_size, hidden_dim)

        self.visualizatioin_save_path = os.path.join(args.save_path, args.dataset + 'explanation')
        if not os.path.isdir(self.visualizatioin_save_path):
            os.makedirs(self.visualizatioin_save_path)

    def lrp_linear_eps(self, r_out, forward_input, forward_output, weight):
        '''

        :param r_out:  relevance of the output (out_feature,)
        :param forward_input: the output tensor (in_feature, )
        :param forward_output: the input tensor (out_feature, )
        :param weight:  weight tensor shape (out_feature, in_feature)
        :return: r_in (in_feature,)
        '''
        assert r_out.dim() == 1
        assert forward_input.dim() == 1
        assert weight.dim() == 2
        attribution = weight * forward_input #(out_feature, in_feature)
        if type(forward_output) == bool:
            forward_output = torch.matmul(forward_input, weight.transpose(0,1))
            # print('matml', forward_output.size())
        forward_output_eps = self.EPS * forward_output.sign() + forward_output  # Z.sign() returns -1 or 0 or 1

        forward_output_eps.masked_fill_(forward_output_eps == 0, self.EPS)  #(out_feature,)
        # print(forward_output_eps.size())
        attribution_norm = attribution.transpose(0,1) / forward_output_eps #(in_feature, out_feature)
        # print(attribution_norm.size())
        relevance_input = torch.sum(attribution_norm * r_out, dim=-1) #(in_feature,)
        assert relevance_input.size() == forward_input.size()
        torch.cuda.empty_cache()
        return relevance_input

    def lrp_mha(self, alpha, value, r_context, context, head_idx):
        '''

        :param alpha:  shape is num_head, num_pixel
        :param value: shape is  num_pixel hiddendim
        :param r_context: shape is  1, hiddendim
        :param context shape is 1, hiddendim
        :return:
        '''
        num_head = alpha.size(0)
        num_pixel = alpha.size(1)
        num_query = r_context.size(0)  #(should be 1 if we use ht ad the query)
        '''using attention for full dimension'''
        # r_value = torch.zeros_like(value).cuda()
        # attention_weight = alpha[head_idx]
        # for i in range(num_pixel):
        #     r_value[i] = self.lrp_linear_eps(r_out=r_context.squeeze(),
        #                                      forward_input=value[i]*attention_weight[i],
        #                                      forward_output=context.squeeze(),
        #                                      weight=torch.eye(self.model.hidden_dim).cuda())
        '''whole heads'''
        # d_k = self.model.hidden_dim//num_head #(should be 64 if using vgg and 8 heads)
        # r_context = r_context.clone().contiguous().view(num_query, num_head, d_k).transpose(0, 1) #(num_head, num_query, d_k)
        # context = context.clone().contiguous().view(num_query, num_head, d_k).transpose(0,1)  #(num_head, num_query, d_k)
        # value = value.clone().contiguous().view(num_pixel, num_head, d_k).transpose(0, 1) #(num_head, num_pixel, d_k)
        # r_value = torch.zeros_like(value)
        # # print(r_context.size(), context.size(), value.size())
        # for h in range(8):
        #     for i in range(num_pixel):
        #         r = self.lrp_linear_eps(r_out=r_context[h,0],
        #                                         forward_input=value[h,i]*alpha[h,i],
        #                                         forward_output=context[h,0],
        #                                         weight=torch.eye(d_k).cuda())
        #         r_value[h, i] = r
        # r_value = r_value.transpose(0, 1).contiguous().view(num_pixel, self.model.hidden_dim)
        '''spread single head'''
        d_k = self.model.hidden_dim//num_head #(should be 64 if using vgg and 8 heads)
        r_context = r_context.clone().contiguous().view(num_query, num_head, d_k).transpose(0, 1) #(num_head, num_query, d_k)
        context = context.clone().contiguous().view(num_query, num_head, d_k).transpose(0,1)  #(num_head, num_query, d_k)
        value = value.clone().contiguous().view(num_pixel, num_head, d_k).transpose(0, 1) #(num_head, num_pixel, d_k)
        r_value = torch.zeros_like(value)
        # print(r_context.size(), context.size(), value.size())
        for i in range(num_pixel):
            r = self.lrp_linear_eps(r_out=r_context[head_idx,0],
                                            forward_input=value[head_idx,i]*alpha[head_idx,i],
                                            forward_output=context[head_idx,0],
                                            weight=torch.eye(d_k).cuda())
            r_value[head_idx, i] = r
        r_value = r_value.transpose(0, 1).contiguous().view(num_pixel, self.model.hidden_dim)
        # print(r_value[0])
        return r_value

    def preprocess_img(self, img_filepath):
        image_data = Image.open(img_filepath).convert('RGB')
        img = self.img_transform(image_data)
        img = img.unsqueeze(0).cuda()
        return img

    def language_lstm_forward(self, xt, ht_m1, ct_m1):
        z = torch.matmul(self.language_weight_i, xt.squeeze())  #(4*hidden_size, 1)
        z = z + torch.matmul(self.language_weight_h, ht_m1) #(4*hidden_size,1)
        z = z + self.language_bias_i + self.language_bias_i
        z0, z1, z2, z3 = z.chunk(4)
        i = torch.sigmoid(z0)
        f = torch.sigmoid(z1)
        c = f * ct_m1 + i * torch.tanh(z2)
        o = torch.sigmoid(z3)
        ht = o * torch.tanh(c)
        ct = c
        return ht, ct, z2, i, f  #(512, )

    def forward_greedy(self, img_filepath):
        self.img = self.preprocess_img(img_filepath) #(bs, C, H, W)
        self.beam_caption, self.beam_caption_encode = self.model.beam_search(self.img, self.word_map, beam_size=1, max_cap_length=20)
        self.beam_caption_encode = [self.word_map['<start>']] + self.beam_caption_encode
        print(f'the predicted caption of {img_filepath} is "{self.beam_caption[0]}"')
        print(self.beam_caption_encode)
        # perform the forward pass and save the intermediate variables
        self.image_features, self.avg_feature = self.model.img_encoder(self.img)  # (bs, fea_dim, H, W), (bs, fea_dim)
        self.num_pixels = self.image_features.size(-1) * self.image_features.size(-2)
        self.image_feature_proj = self.model.relu(self.model.img_projector(self.image_features))  # (bs, hiddendim, H, W)
        self.image_feature_proj = self.image_feature_proj.contiguous()
        self.image_feature_proj = self.image_feature_proj.view(1, self.model.hidden_dim, -1)  # (bs, hidden_dim, num_pixel)
        self.image_feature_proj = self.image_feature_proj.transpose(1,2) #(bs, num_pixel, hidden_dim)
        self.global_img_feature = self.image_feature_proj.mean(1) #(bs, hidden_dim)
        self.caption_length = 50
        self.predictions = torch.zeros(self.caption_length, self.vocab_size).cuda()
        self.xt = torch.zeros(self.caption_length, self.model.embed_dim + self.model.hidden_dim).cuda()
        self.alphas = torch.zeros(self.caption_length, self.num_head, self.num_pixels).cuda()
        self.ht = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.ct = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.gt = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.it_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.ft_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context_aoa_gate = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context_aoa_linear = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context_aoa = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.key = self.model.decoder_k_proj(self.image_feature_proj)  # batch_size, num_pixel, hiddendim
        self.value = self.model.decoder_v_proj(self.image_feature_proj)
        caption = [self.word_map['<start>']]
        for t in range(50):
            it = torch.LongTensor([caption[t]]).cuda()
            word_embedding = self.model.embedding(it)  # (1, embed_dim)
            if self.global_img_feature.dim() == 1:
                self.global_img_feature = self.global_img_feature.unsqueeze(0)
            ht_m1 = self.ht[t]
            ct_m1 = self.ct[t]
            x_t = torch.cat((word_embedding, self.global_img_feature), dim=-1)   # (1, 2*embed_dim)
            # print(x_t.size())
            h_t, c_t, g_t, i_t_act, f_t_act = self.language_lstm_forward(x_t, ht_m1, ct_m1)
            context_t, alpha_t = self.model.decoder_multihead_attention(h_t.unsqueeze(0), self.key, self.value)  # (1, hidden_dim) alpha: (1, num_head, 1, num_pixel)
            context_aoa_gate_t = self.model.decoder_aoa_linear_gate(h_t.unsqueeze(0))
            context_aoa_linear_t = self.model.decoder_aoa_linear(context_t)
            context_aoa_t = torch.sigmoid(context_aoa_gate_t) * context_aoa_linear_t # (1, hiddendim)
            # print(context_aoa_t.size())
            predict_score_t = self.model.fc(context_aoa_t.squeeze() + h_t)  # (bs, vocab_size)
            label = torch.argmax(predict_score_t)
            if label == self.word_map['<end>']:
                sen_idx = [int(w) for w in caption]
                sentence = [' '.join([self.rev_word_map[sen_idx[i]] for i in range(1,len(sen_idx))])]
                print(caption)
                print(sentence)
                print(self.alphas.sum())
                return
            caption.append(label)
            # here we save the intermediate states for further relevance backpropagation
            self.xt[t] = x_t.squeeze()
            self.predictions[t, :] = predict_score_t
            self.alphas[t] = alpha_t.squeeze()
            self.ht[t + 1] = h_t
            self.ct[t + 1] = c_t
            self.gt[t] = g_t
            self.it_act[t] = i_t_act
            self.ft_act[t] = f_t_act
            self.context[t] = context_t.squeeze()
            self.context_aoa_linear[t] = context_aoa_linear_t.squeeze()
            self.context_aoa_gate[t] = context_aoa_gate_t.squeeze()
            self.context_aoa[t] = context_aoa_t.squeeze()

    def teacherforce_forward(self, img, beam_caption_encode):
        # print(beam_caption_encode)
        # perform the forward pass and save the intermediate variables
        image_features, avg_feature = self.model.img_encoder(img)  # (bs, fea_dim, H, W), (bs, fea_dim)
        num_pixels = self.image_features.size(-1) * self.image_features.size(-2)
        image_feature_proj = self.model.relu(self.model.img_projector(image_features))  # (bs, hiddendim, H, W)
        image_feature_proj = image_feature_proj.contiguous()
        image_feature_proj = image_feature_proj.view(1, self.model.hidden_dim, -1)  # (bs, hidden_dim, num_pixel)
        image_feature_proj = image_feature_proj.transpose(1, 2)  # (bs, num_pixel, hidden_dim)
        global_img_feature = image_feature_proj.mean(1)  # (bs, hidden_dim)
        caption_length = len(beam_caption_encode)
        predictions = torch.zeros(caption_length, self.vocab_size).cuda()
        ht = torch.zeros(caption_length + 1, self.model.hidden_dim).cuda()
        ct = torch.zeros(caption_length + 1, self.model.hidden_dim).cuda()
        key = self.model.decoder_k_proj(image_feature_proj)  # batch_size, num_pixel, hiddendim
        value = self.model.decoder_v_proj(image_feature_proj)
        for t in range(len(beam_caption_encode)):
            it = torch.LongTensor([beam_caption_encode[t]]).cuda()
            word_embedding = self.model.embedding(it)  # (1, embed_dim)
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            ht_m1 = ht[t]
            ct_m1 = ct[t]
            x_t = torch.cat((word_embedding, global_img_feature), dim=-1)  # (1, 2*embed_dim)
            # print(x_t.size())
            h_t, c_t, g_t, i_t_act, f_t_act = self.language_lstm_forward(x_t, ht_m1, ct_m1)
            context_t, alpha_t = self.model.decoder_multihead_attention(h_t.unsqueeze(0), key,value)  # (1, hidden_dim) alpha: (1, num_head, 1, num_pixel)
            context_aoa_gate_t = self.model.decoder_aoa_linear_gate(h_t.unsqueeze(0))
            context_aoa_linear_t = self.model.decoder_aoa_linear(context_t)
            context_aoa_t = torch.sigmoid(context_aoa_gate_t) * context_aoa_linear_t  # (1, hiddendim)
            # print(context_aoa_t.size())
            predict_score_t = self.model.fc(context_aoa_t.squeeze() + h_t)  # (bs, vocab_size)
            # here we save the intermediate states for further relevance backpropagation
            predictions[t, :] = predict_score_t
            ht[t + 1] = h_t
            ct[t + 1] = c_t
        return predictions

    def get_hidden_parameters(self, img_filepath):
        self.img = self.preprocess_img(img_filepath)  # (bs, C, H, W)
        self.beam_caption, self.beam_caption_encode = self.model.beam_search(self.img, self.word_map, beam_size=3,
                                                                             max_cap_length=20) # pure sentence without <start> <end>
        self.caption_length = len(self.beam_caption_encode)
        self.beam_caption_encode = [self.word_map['<start>']] + self.beam_caption_encode  # add the start simbol
        print(f'the predicted caption of {img_filepath} is "{self.beam_caption[0]}"')
        print(self.beam_caption_encode)
        # perform the forward pass and save the intermediate variables
        self.image_features, self.avg_feature = self.model.img_encoder(self.img)  # (bs, fea_dim, H, W), (bs, fea_dim)
        self.num_pixels = self.image_features.size(-1) * self.image_features.size(-2)
        self.image_feature_proj_before_act = self.model.img_projector(self.image_features)
        self.image_feature_proj = self.model.relu(self.image_feature_proj_before_act)  # (bs, hiddendim, H, W)
        self.image_feature_proj = self.image_feature_proj.contiguous()
        self.image_feature_proj = self.image_feature_proj.view(1, self.model.hidden_dim, -1)  # (bs, hidden_dim, num_pixel)
        self.image_feature_proj = self.image_feature_proj.transpose(1, 2)  # (bs, num_pixel, hidden_dim)
        self.image_feature_proj_before_act = self.image_feature_proj_before_act.contiguous().view(1,self.model.hidden_dim,-1).transpose(1, 2)
        self.global_img_feature = self.image_feature_proj.mean(1)  # (bs, hidden_dim)
        self.key = self.model.decoder_k_proj(self.image_feature_proj)  # batch_size, num_pixel, hiddendim
        self.value = self.model.decoder_v_proj(self.image_feature_proj)
        self.predictions = torch.zeros(self.caption_length, self.vocab_size).cuda()
        self.xt = torch.zeros(self.caption_length, self.model.embed_dim + self.model.hidden_dim).cuda()  #(start, word_0....word_{capLength-1})
        self.alphas = torch.zeros(self.caption_length, self.num_head, self.num_pixels).cuda()
        self.ht = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.ct = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.gt = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.it_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.ft_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context_aoa = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context_aoa_gate = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context_aoa_linear = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()

        for t in range(self.caption_length):
            it = torch.LongTensor([self.beam_caption_encode[t]]).cuda()
            word_embedding = self.model.embedding(it)  # (1, embed_dim)
            if self.global_img_feature.dim() == 1:
                self.global_img_feature = self.global_img_feature.unsqueeze(0)
            ht_m1 = self.ht[t]
            ct_m1 = self.ct[t]
            x_t = torch.cat((word_embedding, self.global_img_feature), dim=-1)  # (1, 2*embed_dim)
            # print(x_t.size())
            h_t, c_t, g_t, i_t_act, f_t_act = self.language_lstm_forward(x_t, ht_m1, ct_m1)
            context_t, alpha_t = self.model.decoder_multihead_attention(h_t.unsqueeze(0), self.key, self.value)  # (1, hidden_dim) alpha: (1, num_head, 1, num_pixel)
            context_aoa_gate_t = self.model.decoder_aoa_linear_gate(h_t.unsqueeze(0))
            context_aoa_linear_t = self.model.decoder_aoa_linear(context_t)
            context_aoa_t = torch.sigmoid(context_aoa_gate_t) * context_aoa_linear_t # (1, hiddendim)
            # print(context_aoa_t.size()) (1, hidden_dim)   [context_t, ht] 1, 2*hidden_size, decoder_aoa_liner_weight (2*hidden_size, 2*hidden_size)
            predict_score_t = self.model.fc(context_aoa_t + h_t)  # (bs, vocab_size)
            # here we save the intermediate states for further relevance backpropagation
            # print(predict_score_t.size(), x_t.size(),alpha_t.size(), context_t.size(), context_aoa_t.size(), decoder_linear_output.size())
            # print(h_t.size(), c_t.size(), g_t.size(), i_t_act.size(), f_t_act.size())
            self.xt[t] = x_t.squeeze()
            # print(x_t.size())
            self.predictions[t] = predict_score_t.squeeze()
            # print(predict_score_t.size())
            self.alphas[t] = alpha_t.squeeze()
            # print(alpha_t.size())
            self.ht[t + 1] = h_t
            self.ct[t + 1] = c_t
            # print(c_t.size())
            self.gt[t] = g_t
            # print(g_t.size())
            self.it_act[t] = i_t_act
            # print(i_t_act.size())
            self.ft_act[t] = f_t_act
            # print(f_t_act.size())
            self.context[t] = context_t.squeeze()
            # print(context_t.size())
            self.context_aoa[t] = context_aoa_t.squeeze()
            # print(context_aoa_t.size())
            self.context_aoa_linear[t] = context_aoa_linear_t.squeeze()
            self.context_aoa_gate[t] = context_aoa_gate_t.squeeze()

    def explain_caption_wordt(self, t, head_idx):

        assert t < self.caption_length  #(t starts from 0)
        preceeding_cap_length = t+1
        # print(t)
        target_word_encode = self.beam_caption_encode[t+1]
        words = self.beam_caption[0].split(' ')

        language_weight_ig = self.language_weight_i.chunk(4,0)[2]  #(hidden_dim, embed_dim + hidden_dim)
        language_weight_hg = self.language_weight_h.chunk(4,0)[2]  #(hidden_dim, hidden_dim)
        language_weight_g = torch.cat((language_weight_ig, language_weight_hg), dim=-1) #(hidden_dim, 2 * hidden_dim + embed_dim)
        xht = torch.cat((self.xt[:preceeding_cap_length], self.ht[:preceeding_cap_length]), dim=1) #(preceeding_length, 3*hidden_dim)
        # print(xht.size())
        predict_score_t = self.predictions[t] #(vocat_size,)
        # print(target_word_encode, words[t], t + 1, torch.argmax(predict_score_t), predict_score_t[target_word_encode])
        image_feature = self.image_features.view(1, self.model.encoder_raw_dim, self.num_pixels)
        image_feature = image_feature.transpose(1,2).squeeze(0) #(num_pixel, encode_raw_dim)
        word_relevance = torch.zeros(self.vocab_size).cuda()
        word_relevance[target_word_encode] = predict_score_t[target_word_encode]
        self.r_ht = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        self.r_ct = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        self.r_xht = torch.zeros(preceeding_cap_length, self.model.embed_dim + 2 * self.model.hidden_dim).cuda()
        self.r_global_img_feature = torch.zeros(self.model.hidden_dim).cuda()
        self.r_word_embedding = torch.zeros(preceeding_cap_length, self.model.embed_dim).cuda()
        self.r_img_feature = torch.zeros(self.num_pixels, self.model.encoder_raw_dim).cuda()
        self.r_img_feature_proj = torch.zeros(self.num_pixels, self.model.hidden_dim).cuda()
        self.r_context = torch.zeros(self.model.hidden_dim).cuda()
        self.r_context_aoa = torch.zeros(self.model.hidden_dim).cuda()
        r_h2t_context_aoa = self.lrp_linear_eps(r_out=word_relevance,
                                           forward_input=self.ht[t+1]+self.context_aoa[t],
                                           forward_output=predict_score_t,
                                           weight=self.output_weight)
        self.r_ht[t+1] = self.lrp_linear_eps(r_out=r_h2t_context_aoa,
                                             forward_input=self.ht[t+1],
                                             forward_output=self.ht[t+1]+self.context_aoa[t],
                                             weight=torch.eye(self.model.hidden_dim).cuda())

        self.r_context_aoa += self.lrp_linear_eps(r_out=r_h2t_context_aoa,
                                                    forward_input=self.context_aoa[t],
                                                    forward_output=self.ht[t+1]+self.context_aoa[t],
                                                    weight=torch.eye(self.model.hidden_dim).cuda())


        self.r_context = self.lrp_linear_eps(r_out=self.r_context_aoa,
                                           forward_input=self.context[t],
                                           forward_output=self.context_aoa_linear[t],
                                           weight=self.model.decoder_aoa_linear.weight)

        self.r_value = self.lrp_mha(self.alphas[t], self.value.squeeze(), self.r_context.unsqueeze(0),
                                     self.context[t].unsqueeze(0), head_idx)

        for i in range(t+1)[::-1]:
            self.r_ct[i + 1] = self.r_ht[i + 1]
            r_gt = self.lrp_linear_eps(r_out=self.r_ct[i + 1],
                                       forward_input=self.it_act[i] * torch.tanh(self.gt[i]),
                                       forward_output=self.ct[i+1],
                                       weight=torch.eye(self.model.hidden_dim).cuda())
            self.r_ct[i] = self.lrp_linear_eps(r_out=self.r_ct[i + 1],
                                                forward_input=self.ft_act[i] * self.ct[i],
                                                forward_output=self.ct[i+1],
                                                weight=torch.eye(self.model.hidden_dim).cuda())
            self.r_xht[i] = self.lrp_linear_eps(r_out=r_gt,
                                                forward_input=xht[i],
                                                forward_output=self.gt[i],
                                                weight=language_weight_g)
            self.r_ht[i] = self.r_xht[i][self.model.hidden_dim+self.model.embed_dim:]
            self.r_word_embedding[i] = self.r_xht[i][:self.model.embed_dim]
            # if i == t:
            #     self.r_global_img_feature += self.r_xht[i][self.model.embed_dim:self.model.embed_dim+self.model.hidden_dim]
            self.r_global_img_feature += self.r_xht[i][self.model.embed_dim:self.model.embed_dim + self.model.hidden_dim]

        for i in range(self.num_pixels):
            self.r_img_feature_proj[i] = self.lrp_linear_eps(r_out=self.r_global_img_feature,
                                                        forward_input=self.image_feature_proj.squeeze()[i]/self.num_pixels,
                                                        forward_output=self.global_img_feature.squeeze(),
                                                        weight=torch.eye(self.model.hidden_dim).cuda())
            # print(self.value.size())
            self.r_img_feature_proj[i] += self.lrp_linear_eps(r_out=self.r_value[i],
                                                              forward_input=self.image_feature_proj.squeeze()[i],
                                                              forward_output=self.value.squeeze()[i],
                                                              weight=self.model.decoder_v_proj.weight)
            self.r_img_feature[i] = self.r_img_feature[i] + self.lrp_linear_eps(r_out=self.r_img_feature_proj[i],
                                                                                forward_input=image_feature[i],
                                                                                forward_output=self.image_feature_proj_before_act.squeeze()[i],
                                                                                weight=self.model.img_projector.weight.squeeze(-1).squeeze(-1))
        r_words = torch.sum(self.r_word_embedding, dim=-1)
        r_img_feature = self.r_img_feature.unsqueeze(0).transpose(1,2).view(self.image_features.size())
        # print(torch.sum(r_img_feature>0), torch.sum(r_img_feature==0), torch.sum(r_img_feature<0))
        max_abs_r_words = torch.max(torch.abs(r_words))
        if max_abs_r_words > 0:
            r_words = r_words / max_abs_r_words
        torch.cuda.empty_cache()
        return r_img_feature, r_words

    def explain_cnn(self, r_img_feature):
        # cnn_encoder = copy.deepcopy(self.model.img_encoder.encoder)
        relevance_img = self.model.img_encoder.encoder.compute_lrp(self.img, target=r_img_feature)
        # print(torch.sum(relevance_img > 0), torch.sum(relevance_img == 0), torch.sum(relevance_img < 0))
        self.model.img_encoder.encoder.zero_grad()
        return relevance_img

    def explain_caption(self, img_filepath, head_idx, t_list=None):
        self.img_filepath = img_filepath
        self.get_hidden_parameters(img_filepath)
        relevance_imgs = []
        relevance_preceeding_words = []
        lrp_wrapper.add_lrp(self.model.img_encoder.encoder)
        for t in range(self.caption_length):
            with torch.no_grad():
                relevance_img_feature, r_words = self.explain_caption_wordt(t, head_idx)
            relevance_img = self.explain_cnn(relevance_img_feature)
            relevance_imgs.append(relevance_img)
            relevance_preceeding_words.append(r_words)
        assert len(relevance_imgs) == self.caption_length
        self.visualize_explanations(relevance_imgs, head_idx, t=t_list)
        self.save_linguistic_explanation(relevance_preceeding_words)
        torch.cuda.empty_cache()
        return relevance_imgs, relevance_preceeding_words

    def explain_caption_words(self, img_filepath):
        self.img_filepath = img_filepath
        self.get_hidden_parameters(img_filepath)
        relevance_preceeding_words = []
        lrp_wrapper.add_lrp(self.model.img_encoder.encoder)
        for t in range(self.caption_length):
            with torch.no_grad():
                relevance_img_feature, r_words = self.explain_caption_wordt(t, 0)
            relevance_preceeding_words.append(r_words)
        assert len(relevance_preceeding_words) == self.caption_length
        torch.cuda.empty_cache()
        return relevance_preceeding_words


    def save_linguistic_explanation(self, relevance_preceeding_words):
        img_filename = self.img_filepath.split('/')[-1]
        save_dir = os.path.join(self.visualizatioin_save_path, img_filename.strip('.jpg'))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        linguistic_explanation = []
        words = ['<start>'] + self.beam_caption[0].split(' ') + ['<end>']
        for t in range(self.caption_length):
            explanation = []
            relevance_word_t = relevance_preceeding_words[t]
            for i in range(len(relevance_word_t)):
                explanation.append({words[i]:relevance_word_t[i].item()})
            linguistic_explanation.append({words[t+1]: explanation})
        with open(os.path.join(save_dir, self.EX_TYPE + '_linguistic_explanation.yaml'), 'w') as f:
            yaml.safe_dump(linguistic_explanation, f)
            f.close()

    def visualize_explanations(self, relevance_imgs, head_idx, t=None):
        img_filename = self.img_filepath.split('/')[-1]
        save_dir = os.path.join(self.visualizatioin_save_path, img_filename.strip('.jpg'))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        img_original = Image.open(self.img_filepath)
        img_original = img_original.resize((self.args.height, self.args.width))
        img_original.save(os.path.join(save_dir, img_filename))
        x = int(np.sqrt(self.caption_length))
        y = int(np.ceil(self.caption_length / x))
        _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20, 20))
        words = self.beam_caption[0].split(' ')
        axes = axes.flatten()
        assert len(words) == self.caption_length
        for i in range(self.caption_length):
            relevance_img = relevance_imgs[i]
            hm = relevance_img.permute(0,2,3,1).detach().cpu().numpy()
            hm = LRPutil.gamma(hm)
            hm = LRPutil.heatmap(hm)[0]
            hm = Image.fromarray(np.uint8(hm*255))
            hm_show = Image.blend(img_original, hm, 1)
            if isinstance(t, list) and  i in t:
                hm_show.save(os.path.join(save_dir, str(head_idx) + '_lrp_' + words[i] + '.jpg'))
            axes[i].set_title(words[i], fontsize=18)
            axes[i].imshow(hm_show)
        plt.savefig(os.path.join(save_dir, str(head_idx) + '_lrp_hm.jpg'))
        # plt.show()
        _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20, 20))
        axes = axes.flatten()
        for i in range(self.caption_length):
            attention = self.alphas[i][head_idx]
            # print(attention.size())
            # attention = torch.softmax(attention, dim=-1)
            atten_hm = LRPutil.visuallize_attention(img_original, attention,
                                                    (int(np.sqrt(self.num_pixels)),int(np.sqrt(self.num_pixels))), upscale=16)
            if isinstance(t, list) and i in t:
                atten_hm.save(os.path.join(save_dir, str(head_idx) + '_attention_' + words[i] + '.jpg'))
            axes[i].set_title(words[i], fontsize=18)
            axes[i].imshow(atten_hm)

        plt.savefig(os.path.join(save_dir, str(head_idx) + '_attention_hm.jpg'))
        # plt.show()


class ExplainAOAGradient(object):
    EX_TYPE = 'gradient'
    def __init__(self, args, word_map):
        super(ExplainAOAGradient, self).__init__()
        self.args = args
        self.word_map = word_map
        self.rev_word_map = {v: k for k, v in word_map.items()}
        self.vocab_size = len(word_map)
        self.num_head = args.num_head
        self.model = AOAModel(args.embed_dim, args.hidden_dim, args.num_head, len(word_map), args.encoder)
        checkpoint = torch.load(args.weight)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.cuda()
        self.model.eval()

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.img_transform = transforms.Compose([transforms.Resize(size=(args.height, args.width)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=self.mean, std=self.std)])

        self.language_weight_i = self.model.LanguageLSTM.weight_ih #(4*hidden_size, hiddendim+embed_dim)
        self.language_weight_h = self.model.LanguageLSTM.weight_hh #(4*hidden_size, hiddendim)
        self.language_bias_i = self.model.LanguageLSTM.bias_ih #(4*hidden_size,)
        self.language_bias_h = self.model.LanguageLSTM.bias_hh #(4*hidden_size,)

        self.output_weight = self.model.fc.weight  #(vocab_size, hidden_dim)

        self.visualizatioin_save_path = os.path.join(args.save_path, args.dataset + 'explanation')
        if not os.path.isdir(self.visualizatioin_save_path):
            os.makedirs(self.visualizatioin_save_path)

    def preprocess_img(self, img_filepath):
        image_data = Image.open(img_filepath).convert('RGB')
        img = self.img_transform(image_data)
        img = img.unsqueeze(0).cuda()
        return img

    def language_lstm_forward(self, xt, ht_m1, ct_m1):
        z = torch.matmul(self.language_weight_i, xt.squeeze(0))  #(4*hidden_size, )
        z = z + torch.matmul(self.language_weight_h, ht_m1) #(4*hidden_size,)
        z = z + self.language_bias_h + self.language_bias_i
        z0, z1, z2, z3 = z.chunk(4)
        i = torch.sigmoid(z0)
        f = torch.sigmoid(z1)
        g = torch.tanh(z2)
        c = f * ct_m1 + i * torch.tanh(z2)
        o = torch.sigmoid(z3)
        ht = o * torch.tanh(c)
        ct = c
        return ht, ct, z0, z1, z2, z3, i, f, g, o

    def get_hidden_parameters(self, img_filepath):
        self.img = self.preprocess_img(img_filepath)  # (bs, C, H, W)
        self.beam_caption, self.beam_caption_encode = self.model.beam_search(self.img, self.word_map, beam_size=3,
                                                                             max_cap_length=20)
        self.beam_caption_encode = [self.word_map['<start>']] + self.beam_caption_encode
        print(f'the predicted caption of {img_filepath} is "{self.beam_caption[0]}"')
        print(self.beam_caption_encode)
        # perform the forward pass and save the intermediate variables
        self.image_features, self.avg_feature = self.model.img_encoder(self.img)  # (bs, fea_dim, H, W), (bs, fea_dim)
        self.num_pixels = self.image_features.size(-1) * self.image_features.size(-2)
        self.image_feature_proj = self.model.relu(self.model.img_projector(self.image_features))  # (bs, hiddendim, H, W)
        self.image_feature_proj = self.image_feature_proj.contiguous()
        self.image_feature_proj = self.image_feature_proj.view(1, self.model.hidden_dim, -1)  # (bs, hidden_dim, num_pixel)
        self.image_feature_proj = self.image_feature_proj.transpose(1, 2)  # (bs, num_pixel, hidden_dim)
        self.global_img_feature = self.image_feature_proj.mean(1)  # (bs, hidden_dim)
        self.caption_length = len(self.beam_caption_encode) -1
        self.predictions = torch.zeros(self.caption_length, self.vocab_size).cuda()
        self.xt = torch.zeros(self.caption_length, self.model.embed_dim + self.model.hidden_dim).cuda()
        self.alphas = torch.zeros(self.caption_length, self.num_head, self.num_pixels).cuda()
        self.ht = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.ct = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.it = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.ft = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.gt = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.ot = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.it_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.ft_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.gt_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.ot_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context_aoa_gate = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context_aoa_linear = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context_aoa = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.key = self.model.decoder_k_proj(self.image_feature_proj)  # batch_size, num_pixel, hiddendim
        self.value = self.model.decoder_v_proj(self.image_feature_proj)
        for t in range(self.caption_length):
            it = torch.LongTensor([self.beam_caption_encode[t]]).cuda()
            word_embedding = self.model.embedding(it)  # (1, embed_dim)
            if self.global_img_feature.dim() == 1:
                self.global_img_feature = self.global_img_feature.unsqueeze(0)
            ht_m1 = self.ht[t]
            ct_m1 = self.ct[t]
            x_t = torch.cat((word_embedding, self.global_img_feature), dim=-1)  # (1, 2*embed_dim)
            h_t, c_t, i_t, f_t, g_t, o_t, i_t_act, f_t_act, g_t_act, o_t_act = self.language_lstm_forward(x_t, ht_m1, ct_m1)
            context_t, alpha_t = self.model.decoder_multihead_attention(h_t.unsqueeze(0), self.key, self.value)  # (1, hidden_dim) alpha: (1, num_head, 1, num_pixel)
            context_aoa_gate_t = self.model.decoder_aoa_linear_gate(h_t.unsqueeze(0))
            context_aoa_linear_t = self.model.decoder_aoa_linear(context_t)
            context_aoa_t = torch.sigmoid(context_aoa_gate_t) * context_aoa_linear_t # (1, hiddendim)
            predict_score_t = self.model.fc(context_aoa_t.squeeze() + h_t)  # (bs, vocab_size)
            # here we save the intermediate states for further relevance backpropagation
            self.xt[t] = x_t.squeeze()
            self.predictions[t, :] = predict_score_t.squeeze()
            self.alphas[t] = alpha_t.squeeze()
            self.ht[t + 1] = h_t
            self.ct[t + 1] = c_t
            self.it[t] = i_t
            self.ft[t] = f_t
            self.gt[t] = g_t
            self.ot[t] = o_t
            self.it_act[t] = i_t_act
            self.ft_act[t] = f_t_act
            self.gt_act[t] = g_t_act
            self.ot_act[t] = o_t_act
            self.context[t] = context_t.squeeze()
            self.context_aoa[t] = context_aoa_t.squeeze()
            self.context_aoa_linear[t] = context_aoa_linear_t.squeeze()
            self.context_aoa_gate[t] = context_aoa_gate_t.squeeze()

    def teacherforce_forward(self, img, beam_caption_encode):
        # print(beam_caption_encode)
        # perform the forward pass and save the intermediate variables
        image_features, avg_feature = self.model.img_encoder(img)  # (bs, fea_dim, H, W), (bs, fea_dim)
        num_pixels = self.image_features.size(-1) * self.image_features.size(-2)
        image_feature_proj = self.model.relu(self.model.img_projector(image_features))  # (bs, hiddendim, H, W)
        image_feature_proj = image_feature_proj.contiguous()
        image_feature_proj = image_feature_proj.view(1, self.model.hidden_dim, -1)  # (bs, hidden_dim, num_pixel)
        image_feature_proj = image_feature_proj.transpose(1, 2)  # (bs, num_pixel, hidden_dim)
        global_img_feature = image_feature_proj.mean(1)  # (bs, hidden_dim)
        caption_length = len(beam_caption_encode)
        predictions = torch.zeros(caption_length, self.vocab_size).cuda()
        ht = torch.zeros(caption_length + 1, self.model.hidden_dim).cuda()
        ct = torch.zeros(caption_length + 1, self.model.hidden_dim).cuda()
        key = self.model.decoder_k_proj(image_feature_proj)  # batch_size, num_pixel, hiddendim
        value = self.model.decoder_v_proj(image_feature_proj)
        for t in range(len(beam_caption_encode)):
            it = torch.LongTensor([beam_caption_encode[t]]).cuda()
            word_embedding = self.model.embedding(it)  # (1, embed_dim)
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            ht_m1 = ht[t]
            ct_m1 = ct[t]
            x_t = torch.cat((word_embedding, global_img_feature), dim=-1)  # (1, 2*embed_dim)
            # print(x_t.size())
            h_t, c_t, i_t, f_t, g_t, o_t, i_t_act, f_t_act, g_t_act, o_t_act = self.language_lstm_forward(x_t, ht_m1, ct_m1)
            context_t, alpha_t = self.model.decoder_multihead_attention(h_t.unsqueeze(0), key,value)  # (1, hidden_dim) alpha: (1, num_head, 1, num_pixel)
            context_aoa_gate_t = self.model.decoder_aoa_linear_gate(h_t.unsqueeze(0))
            context_aoa_linear_t = self.model.decoder_aoa_linear(context_t)
            context_aoa_t = torch.sigmoid(context_aoa_gate_t) * context_aoa_linear_t  # (1, hiddendim)
            # print(context_aoa_t.size())
            predict_score_t = self.model.fc(context_aoa_t.squeeze() + h_t)  # (bs, vocab_size)
            # here we save the intermediate states for further relevance backpropagation
            predictions[t, :] = predict_score_t
            ht[t + 1] = h_t
            ct[t + 1] = c_t
        return predictions

    def gradient_mha(self, d_context, alpha, head_idx):
        '''

        :param d_context: (hidden_dim)
        :param alpha:  (num_head, num_pixel)
        :return:
        '''
        num_head = alpha.size(0)
        num_pixel = alpha.size(1)
        num_query = d_context.size(0)  #(should be 1 if we use ht ad the query)
        d_k = self.model.hidden_dim // num_head  # (should be 64 if using vgg and 8 heads)
        d_context = d_context.clone().contiguous().view(num_query, num_head, d_k).transpose(0, 1)  # (num_head, num_query, d_k)
        alpha = alpha.unsqueeze(1)  # (num_head, num_query, num_pixel)
        d_value = torch.zeros(num_head, num_pixel, d_k).cuda()
        for i in range(head_idx, head_idx+1):
            for j in range(num_pixel):
                d_value[i,j] = d_context[i,0]*alpha[i,0,j]
        d_value = d_value.transpose(0, 1).contiguous().view(num_pixel, self.model.hidden_dim)
        return d_value

    def explain_caption_wordt(self, t, head_idx):
        assert t < self.caption_length  #(t starts from 0)
        preceeding_cap_length = t+1
        target_word_encode = self.beam_caption_encode[t+1]
        d_word_pred = torch.zeros(1, self.vocab_size).cuda()
        d_word_pred[0, target_word_encode] = 1.  #(1, vocab_size)

        d_ht = torch.zeros(preceeding_cap_length + 1, self.model.hidden_dim).cuda()
        d_ct = torch.zeros(preceeding_cap_length + 1, self.model.hidden_dim).cuda()
        d_it = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_ft = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_gt = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_ot = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_it_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_ft_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_gt_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_ot_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_xt = torch.zeros(preceeding_cap_length, self.model.hidden_dim + self.model.embed_dim).cuda()
        d_global_img_feature = torch.zeros(self.model.hidden_dim).cuda()
        d_word_embedding = torch.zeros(preceeding_cap_length, self.model.embed_dim).cuda()
        d_img_feature = torch.zeros(self.num_pixels, self.model.encoder_raw_dim).cuda()
        d_context_aoa = torch.zeros(self.model.hidden_dim).cuda()
        d_decoder_A = torch.zeros(self.model.hidden_dim).cuda()
        d_decoder_B = torch.zeros(self.model.hidden_dim).cuda()
        d_value = torch.zeros(self.num_pixels, self.model.hidden_dim).cuda()

        # here we start the backward
        d_context_aoa_ht = torch.matmul(d_word_pred, self.output_weight).squeeze()
        d_context_aoa += d_context_aoa_ht * 1
        d_ht[t + 1] = d_context_aoa_ht * 1

        d_decoder_A += d_context_aoa * torch.sigmoid(self.context_aoa_gate[t])
        d_decoder_B += d_context_aoa* self.context_aoa_linear[t] * (
                    1 - torch.sigmoid(self.context_aoa_gate[t])) * torch.sigmoid(self.context_aoa_gate[t])
        d_context_t = torch.matmul(d_decoder_A, self.model.decoder_aoa_linear.weight).squeeze()  # (hidden_dim * 2)
        d_ht[t + 1] += torch.matmul(d_decoder_B, self.model.decoder_aoa_linear_gate.weight).squeeze()
        # d_context_t = d_context_ht[:self.model.hidden_dim]
        d_value += self.gradient_mha(d_context_t.unsqueeze(0), self.alphas[t].squeeze(), head_idx)
        # d_ht[t + 1] += d_context_ht[self.model.hidden_dim:]
        for i in range(preceeding_cap_length)[::-1]:
            d_ot_act[i] = d_ht[i+1] * torch.tanh(self.ct[i+1])
            d_ct[i+1] = d_ct[i+1] + d_ht[i+1] * self.ot_act[i] * (1-(torch.tanh(self.ct[i+1]))**2)
            d_ft_act[i] = d_ct[i+1] * self.ct[i]
            d_ct[i] = d_ct[i+1] * self.ft_act[i]
            d_it_act[i] = d_ct[i+1] * self.gt_act[i]
            d_gt_act[i] = d_ct[i+1] * self.it_act[i]
            d_it[i] = d_it_act[i] * self.it_act[i] * (1 - self.it_act[i])
            d_ft[i] = d_ft_act[i] * self.ft_act[i] * (1 - self.ft_act[i])
            d_ot[i] = d_ot_act[i] * self.ot_act[i] * (1 - self.ot_act[i])
            d_gt[i] = d_gt_act[i] * (1 - (self.gt_act[i]) ** 2)
            d_gates = torch.cat((d_it[i: i+1], d_ft[i: i+1], d_gt[i: i+1], d_ot[i: i+1]),dim=1) #(1, 4*hidden_dim)
            d_ht[i] = torch.matmul(d_gates, self.language_weight_h).squeeze() #(hidden_dim)
            d_xt[i] = torch.matmul(d_gates, self.language_weight_i).squeeze() #(embed_dim + hidden_dim)
            d_global_img_feature = d_xt[i][self.model.embed_dim:]
            d_word_embedding[i] = d_xt[i][:self.model.embed_dim]
        d_img_feature_proj = torch.matmul(d_value,self.model.decoder_v_proj.weight)
        for i in range(self.num_pixels):
            d_img_feature_proj[i] += d_global_img_feature / self.num_pixels
            d_img_feature[i] = torch.matmul(d_img_feature_proj[i], self.model.img_projector.weight.squeeze(-1).squeeze(-1)).squeeze()
        r_words = torch.sum(d_word_embedding, dim=-1)
        max_abs_r_words = torch.max(torch.abs(r_words))
        if max_abs_r_words > 0:
            r_words = r_words / max_abs_r_words
        d_img_feature = d_img_feature.unsqueeze(0).transpose(1,2).view(self.image_features.size()) #(bs, C, H, W)
        return d_img_feature, r_words

    def explain_cnn(self, d_img_feature):
        cnn_encoder = copy.deepcopy(self.model.img_encoder.encoder)
        cnn_encoder.zero_grad()
        sample = copy.deepcopy(self.img)
        sample.requires_grad = True
        sample.retain_grad()
        with torch.enable_grad():
            image_feature = cnn_encoder(sample)
            image_feature.backward(d_img_feature,retain_graph=True)
            # print(sample.grad)
            result = sample.grad.detach().clone()
        cnn_encoder.zero_grad()
        sample.grad.zero_()
        del cnn_encoder
        return result

    def explain_caption(self, img_filepath, head_idx,t_list=None,):
        self.img_filepath = img_filepath
        with torch.no_grad():
            self.get_hidden_parameters(img_filepath)
        self.image_feature_proj = self.image_feature_proj.transpose(1,2)
        relevance_imgs = []
        relevance_preceeding_words = []
        for t in range(self.caption_length):
            with torch.no_grad():
                relevance_img_feature, r_words = self.explain_caption_wordt(t, head_idx)
            relevance_img = self.explain_cnn(relevance_img_feature)
            relevance_imgs.append(relevance_img)
            relevance_preceeding_words.append(r_words)
        torch.cuda.empty_cache()
        assert len(relevance_imgs) == self.caption_length
        self.visualize_explanations(relevance_imgs, head_idx,t=t_list)
        self.save_linguistic_explanation(relevance_preceeding_words)
        return relevance_imgs, relevance_preceeding_words

    def explain_caption_words(self, img_filepath):
        self.img_filepath = img_filepath
        with torch.no_grad():
            self.get_hidden_parameters(img_filepath)
        self.image_feature_proj = self.image_feature_proj.transpose(1,2)
        relevance_preceeding_words = []
        for t in range(self.caption_length):
            with torch.no_grad():
                relevance_img_feature, r_words = self.explain_caption_wordt(t)
            relevance_preceeding_words.append(r_words)
        torch.cuda.empty_cache()
        assert len(relevance_preceeding_words) == self.caption_length
        return relevance_preceeding_words

    def save_linguistic_explanation(self, relevance_preceeding_words):
        img_filename = self.img_filepath.split('/')[-1]
        save_dir = os.path.join(self.visualizatioin_save_path, img_filename.strip('.jpg'))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        linguistic_explanation = []
        words = ['<start>'] + self.beam_caption[0].split(' ')
        for t in range(self.caption_length):
            explanation = []
            relevance_word_t = relevance_preceeding_words[t]
            for i in range(len(relevance_word_t)):
                explanation.append({words[i]:relevance_word_t[i].item()})
            linguistic_explanation.append({words[t+1]: explanation})
        with open(os.path.join(save_dir, self.EX_TYPE + '_linguistic_explanation.yaml'), 'w') as f:
            yaml.safe_dump(linguistic_explanation, f)
            f.close()

    def visualize_explanations(self, relevance_imgs, head_idx, t=None):
        img_filename = self.img_filepath.split('/')[-1]
        save_dir = os.path.join(self.visualizatioin_save_path, img_filename.strip('.jpg'))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        img_original = Image.open(self.img_filepath)
        img_original = img_original.resize((self.args.height, self.args.width))
        img_original.save(os.path.join(save_dir, img_filename))
        x = int(np.sqrt(self.caption_length))
        y = int(np.ceil(self.caption_length / x))
        _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20, 20))
        words = self.beam_caption[0].split(' ')
        axes = axes.flatten()
        assert len(words) == self.caption_length
        for i in range(self.caption_length):
            relevance_img = relevance_imgs[i]
            hm = relevance_img.permute(0,2,3,1).detach().cpu().numpy()
            hm = LRPutil.gamma(hm)
            hm = LRPutil.heatmap(hm)[0]
            if isinstance(t, list) and  i in t:
                hm = Image.fromarray(np.uint8(hm * 255))
                hm.save(os.path.join(save_dir, str(head_idx) + '_gradient_' + words[i] + '.jpg'))
            axes[i].set_title(words[i], fontsize=18)
            axes[i].imshow(hm)
        plt.savefig(os.path.join(save_dir,str(head_idx) + 'gradient_hm.jpg'))


class ExplainAOAGuidedGradient(ExplainAOAGradient):
    EX_TYPE = 'GuidedBackpropagate'
    def register_hooks(self, model):
        def forward_hook_fn(module, input, output):
            module.output_ = output
        def backward_hook_fn(module, grad_in, grad_out):
            grad = module.output_.detach().clone()
            grad[grad>0] = 1
            grad[grad<=0] = 0
            positive_grad_out = torch.clamp(grad_out[0].detach().clone(), min=0.0)
            new_grad_in = positive_grad_out * grad

            return (new_grad_in,)
        modules = list(model.named_children())
        for name, module in modules:
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook_fn)
                module.register_backward_hook(backward_hook_fn)
    def delete_hooks(self, model):
        modules = list(model.named_children())
        for name, module in modules:
            module.zero_grad()
            if isinstance(module, nn.ReLU):
                module.output_ = None
                module.register_forward_hook(None)
                module.register_backward_hook(None)

    def explain_cnn(self, d_img_feature):
        # we performe the guided backpropagate here
        d_img_feature[self.image_features<0] = 0
        cnn_encoder = copy.deepcopy(self.model.img_encoder.encoder)
        cnn_encoder.zero_grad()
        self.register_hooks(cnn_encoder)
        sample = copy.deepcopy(self.img)
        sample.requires_grad = True
        sample.retain_grad()
        with torch.enable_grad():
            image_feature = cnn_encoder(sample)
            image_feature.backward(d_img_feature, retain_graph=True)
            # print(sample.grad)
            result = sample.grad.detach().clone()
        cnn_encoder.zero_grad()
        sample.grad.zero_()
        self.delete_hooks(cnn_encoder)
        del cnn_encoder
        gc.collect()
        return result

    def visualize_explanations(self, relevance_imgs, head_idx, t=None):
        img_filename = self.img_filepath.split('/')[-1]
        save_dir = os.path.join(self.visualizatioin_save_path, img_filename.strip('.jpg'))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        img_original = Image.open(self.img_filepath)
        img_original = img_original.resize((self.args.height, self.args.width))
        img_original.save(os.path.join(save_dir, img_filename))
        x = int(np.sqrt(self.caption_length))
        y = int(np.ceil(self.caption_length / x))
        _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20, 20))
        words = self.beam_caption[0].split(' ')
        axes = axes.flatten()
        assert len(words) == self.caption_length
        for i in range(self.caption_length):
            relevance_img = relevance_imgs[i]
            hm = relevance_img.permute(0,2,3,1).detach().cpu().numpy()
            hm = LRPutil.gamma(hm)
            hm = LRPutil.heatmap(hm)[0]
            if isinstance(t, list) and  i in t:
                hm = Image.fromarray(np.uint8(hm * 255))
                hm.save(os.path.join(save_dir, str(head_idx) + '_GuidedBackpropagate_' + words[i] + '.jpg'))
            axes[i].set_title(words[i], fontsize=18)
            axes[i].imshow(hm)
        plt.savefig(os.path.join(save_dir, str(head_idx) + 'GuidedBackpropagate_hm.jpg'))


class ExplainAOAGradCam(ExplainAOAGradient):
    EX_TYPE = 'GradCam'
    def explain_cnn(self, d_img_feature):
        cam_heatmap = self.grad_cam(self.image_features, d_img_feature) #(H, W)
        cam_heatmap = cam_heatmap.unsqueeze(0) #(1, H* W)
        return cam_heatmap

    def grad_cam(self, img_feature, grads):
        '''
        :param img_feature: bs, fea_dim, h, w
        :param grads: bs, fea_dim, h, w
        :return:
        '''
        weights = torch.mean(grads, dim=(2,3), keepdim=True) #(1,fea_dim,1,1)
        weighted_feature = img_feature * weights
        # print(weighted_feature.size())
        cam = torch.sum(weighted_feature, dim=(0,1)) #(H, W)
        # print(cam.size())
        cam = torch.relu(cam)
        cam_heatmap = cam / (torch.max(torch.abs(cam))+1e-6)
        return cam_heatmap.view(-1)

    def visualize_explanations(self, relevance_imgs, head_idx, t=None):
        img_filename = self.img_filepath.split('/')[-1]
        save_dir = os.path.join(self.visualizatioin_save_path, img_filename.strip('.jpg'))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        img_original = Image.open(self.img_filepath)
        img_original = img_original.resize((self.args.height, self.args.width))
        img_original.save(os.path.join(save_dir, img_filename))
        x = int(np.sqrt(self.caption_length))
        y = int(np.ceil(self.caption_length / x))
        words = self.beam_caption[0].split(' ')
        _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20, 20))
        axes = axes.flatten()
        for i in range(self.caption_length):
            atten_hm = LRPutil.visuallize_attention(img_original, relevance_imgs[i],
                                                    (int(np.sqrt(self.num_pixels)),int(np.sqrt(self.num_pixels))),upscale=16)
            if isinstance(t, list) and  i in t:
                atten_hm.save(os.path.join(save_dir, str(head_idx) + '_GradCam_' + words[i] + '.jpg'))
            axes[i].set_title(words[i], fontsize=18)
            axes[i].imshow(atten_hm)
        plt.savefig(os.path.join(save_dir, str(head_idx) + 'GradCAM_hm.jpg'))


class ExplainAOAGuidedGradCam(ExplainAOAGuidedGradient):
    EX_TYPE = 'GuidedGradCam'
    def grad_cam(self, img_feature, grads):
        '''
        :param img_feature: bs, fea_dim, h, w
        :param grads: bs, fea_dim, h, w
        :return:
        '''
        weights = torch.mean(grads, dim=(2,3), keepdim=True) #(1,fea_dim,1,1)
        weighted_feature = img_feature * weights
        cam = torch.sum(weighted_feature, dim=(0,1)) #(H, W)
        cam = torch.clamp(cam, min=0)
        cam_heatmap = cam / (torch.max(torch.abs(cam))+1e-6)
        return cam_heatmap

    def explain_cnn(self, d_img_feature):
        d_img_feature[self.image_features < 0] = 0
        cnn_encoder = copy.deepcopy(self.model.img_encoder.encoder)
        cnn_encoder.zero_grad()
        self.register_hooks(cnn_encoder)
        sample = copy.deepcopy(self.img)
        sample.requires_grad = True
        sample.retain_grad()
        with torch.enable_grad():
            image_feature = cnn_encoder(sample)
            image_feature.backward(d_img_feature, retain_graph=True)
            guided_gradient = sample.grad.detach().clone()
        cam = self.grad_cam(self.image_features, d_img_feature)
        # cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(self.img.size(2), self.img.size(3)), mode='bilinear', align_corners=True)
        cam = skimage.transform.pyramid_expand(cam.detach().cpu().numpy(), upscale=16,multichannel=False)
        with torch.no_grad():
            guided_results = guided_gradient * torch.from_numpy(cam).cuda().float().expand_as(guided_gradient)
        cnn_encoder.zero_grad()
        sample.grad.zero_()
        self.delete_hooks(cnn_encoder)
        del cnn_encoder
        return guided_results

    def visualize_explanations(self, relevance_imgs, head_idx,t=None):
        img_filename = self.img_filepath.split('/')[-1]
        save_dir = os.path.join(self.visualizatioin_save_path, img_filename.strip('.jpg'))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        img_original = Image.open(self.img_filepath)
        img_original = img_original.resize((self.args.height, self.args.width))
        img_original.save(os.path.join(save_dir, img_filename))
        x = int(np.sqrt(self.caption_length))
        y = int(np.ceil(self.caption_length / x))
        _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20, 20))
        words = self.beam_caption[0].split(' ')
        axes = axes.flatten()
        assert len(words) == self.caption_length
        for i in range(self.caption_length):
            relevance_img = relevance_imgs[i]
            hm = relevance_img.permute(0,2,3,1).detach().cpu().numpy()
            hm = LRPutil.gamma(hm)
            hm = LRPutil.heatmap(hm)[0]
            if isinstance(t, list) and  i in t:
                hm = Image.fromarray(np.uint8(hm * 255))
                hm.save(os.path.join(save_dir, str(head_idx) + '_GuidedGradCam_' + words[i] + '.jpg'))
            axes[i].set_title(words[i], fontsize=18)
            axes[i].imshow(hm)
        plt.savefig(os.path.join(save_dir, str(head_idx) + 'GuidedGradCam_hm.jpg'))

'''models using bottom up features'''
class AOAModelBU(nn.Module):
    '''
    '''
    EPS = LRPutil.EPSILON
    def __init__(self, embed_dim, hidden_dim, num_head, vocab_size, encoder_type):
        super(AOAModelBU, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.encoder_type = encoder_type
        self.vocab_size = vocab_size
        self.num_head = num_head
        if hidden_dim % num_head != 0:
            raise TypeError("the number of head should be dividable by the hidden dim")
        self.dropout = nn.Dropout(0.3)
        # the image encoder to generate image features (bs, C, H, W)
        self.encoder_raw_dim = 2048
        print(f'==========Encoded image feature dim is {self.encoder_raw_dim}==========')
        self.img_projector = nn.Linear(self.encoder_raw_dim, self.hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.LanguageLSTM = nn.LSTMCell(hidden_dim+embed_dim, hidden_dim)
        self.decoder_k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.decoder_v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.decoder_multihead_attention = MultiHeadedDotAttention(num_head=num_head, hidden_dim=hidden_dim, project_k_v_flag=False, norm_q=False, aoa=False)
        self.decoder_aoa_linear_gate = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.decoder_aoa_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()
        # self.refiner_batchnorm = nn.BatchNorm1d(hidden_dim, track_running_stats=True)

    def init_hidden_state(self, V):
        h = torch.zeros(V.shape[0], self.hidden_dim).cuda()
        c = torch.zeros(V.shape[0], self.hidden_dim).cuda()
        return h, c

    def predict_next_word(self,image_feature_proj, xt, states):
        '''
        :param image_feature_proj:  bs, num_pixel, hidden_dim
        :param xt: bs, hidden_dim + embedding dim
        :param states: (ht, ct, context) each with shape bs, hidden_dim
        :return:
        '''
        htm1, ctm1 = states  # (bs, hidden_dim, )
        ht, ct = self.LanguageLSTM(xt, (htm1, ctm1))
        key = self.decoder_k_proj(image_feature_proj) # batch_size, num_pixel, 2 * hiddendim this is the concatenated key and value
        value = self.decoder_v_proj(image_feature_proj)
        context, alpha_t = self.decoder_multihead_attention(ht, key, value) #(bs, hidden_dim) alpha: (bs, num_pixel)
        context_aoa_gate = self.decoder_aoa_linear_gate(ht)
        context_aoa_linear = self.decoder_aoa_linear(context)
        context_aoa =  torch.sigmoid(context_aoa_gate) * context_aoa_linear#(bs, hiddendim)
        predict_score_t = self.fc(self.dropout(context_aoa+ht))  # (bs, vocab_size)
        return predict_score_t, alpha_t, None, (ht, ct)

    def forward(self, images_features, encoded_captions, caption_lengths, ss_prob):
        """
        images: the encoded images from the encoder, of shape (batch_size, C, H, W)
        global_features: the global image features returned by the Encoder, of shape: (batch_size, hidden_dim)
        encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        """
        batch_size = images_features.size(0)
        image_feature_proj = self.relu(self.img_projector(images_features)) # (bs, 36, hiddendim)
        # print(image_feature_proj.size())
        image_feature_proj = image_feature_proj.contiguous() #(bs, num_pixel, hidden_dim)
        global_img_feature = torch.mean(image_feature_proj, dim=1) #(bs, hidden_dim)
        h, c = self.init_hidden_state(image_feature_proj)
        state = (h, c)
        max_length = max(caption_lengths)-1
        predictions = torch.zeros(batch_size, max_length, self.vocab_size).cuda()
        if ss_prob is None:
            ss_flag = False
        else:
            random_num = np.random.uniform(0.0, 1.0, size=(batch_size,))
            ss_mask = random_num < ss_prob
            ss_mask = torch.from_numpy(ss_mask).long().cuda()
            if ss_mask.sum() > 0:
                ss_flag = True
            else:
                ss_flag = False
        for t in range(max_length):
            if t>2 and ss_flag:
                it = last_label*ss_mask + encoded_captions[:, t] * (1-ss_mask)
                word_embedding = self.embedding(it) # (batch_size, embed_dim)
            else:
                # print(t,max_length,encoded_captions)
                word_embedding = self.embedding(encoded_captions[:,t])
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            xt = torch.cat((word_embedding, global_img_feature), dim=-1)   # (batch_size, hidden_dim + embed_dim)
            predict_score_t, alpha_t, beta_t, state = self.predict_next_word(image_feature_proj, xt, state)
            predictions[:, t,:] = predict_score_t
            last_scores = torch.log_softmax(predict_score_t,-1)
            # print(last_scores.size())
            last_label = torch.argmax(last_scores, -1)  #(batch_size, )
            # print(last_label.size())
        return predictions, None, None, last_scores, max_length

    def sample(self, images_features, word_map, caption_lengths, opt={}):

        batch_size = images_features.size(0)
        sample_method = opt.get('sample_method', 'greedy')
        temperature = opt.get('temperature', 1.0)
        max_length = max(caption_lengths) - 1
        image_feature_proj = self.relu(self.img_projector(images_features))  # (bs, 36, hiddendim)
        global_img_feature = torch.mean(image_feature_proj, dim=1)  # (bs, hidden_dim)
        h, c = self.init_hidden_state(image_feature_proj)
        state = (h, c)
        seq = torch.zeros(batch_size,max_length).long().cuda()
        seq_logprobs = torch.zeros(batch_size, max_length).cuda()
        for t in range(max_length):
            if t == 0:
                it = torch.ones(batch_size).long().cuda() * word_map['<start>']
            word_embedding = self.embedding(it)
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            xt = torch.cat((word_embedding,  global_img_feature), dim=-1)  # (batch_size, 2*embed_dim)
            predict_score_t, alpha_t, beta_t, state = self.predict_next_word(image_feature_proj, xt, state)
            predict_score_t = torch.log_softmax(predict_score_t,dim=-1)
            it, sampleLpgprobs = self.sample_next_word(predict_score_t, sample_method, temperature)
            # sample the next word
            if t == 0:
                current_finished = it == word_map['<end>']
                unfinished = ~current_finished
            else:
                current_finished = it == word_map['<end>']
                unfinished = unfinished * ~current_finished
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            seq_logprobs[:, t] = sampleLpgprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break
        return seq, seq_logprobs, max_length

    def sample_next_word(self, logprobs, sample_method, temperature):
        if sample_method == 'greedy':
            sampleLogprobs, it = torch.max(logprobs.detach(), 1)
            it = it.view(-1).long()
        elif sample_method == 'gumbel': # gumbel softmax
            def sample_gumbel(shape, eps=1e-20):
                U = torch.rand(shape).cuda()
                return -torch.log(-torch.log(U + eps) + eps)
            def gumbel_softmax_sample(logits, temperature):
                y = logits + sample_gumbel(logits.size())
                return torch.log_softmax(y / temperature, dim=-1)
            _logprobs = gumbel_softmax_sample(logprobs, temperature)
            _, it = torch.max(_logprobs.data, 1)
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1)) # gather the logprobs at sampled positions
        else:
            logprobs = logprobs / temperature
            if sample_method.startswith == 'top': # topk sampling
                top_num = float(sample_method[3:])
                if 0 < top_num < 1:
                    # nucleus sampling from # The Curious Case of Neural Text Degeneration
                    probs = torch.softmax(logprobs, dim=1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
                    _cumsum = sorted_probs.cumsum(1)
                    mask = _cumsum < top_num
                    mask = torch.cat([torch.ones_like(mask[:,:1]), mask[:,:-1]], 1)
                    sorted_probs = sorted_probs * mask.float()
                    sorted_probs = sorted_probs / sorted_probs.sum(1, keepdim=True)
                    logprobs.scatter_(1, sorted_indices, sorted_probs.log())
                else:
                    the_k = int(top_num)
                    tmp = torch.empty_like(logprobs).fill_(float('-inf'))
                    topk, indices = torch.topk(logprobs, the_k, dim=1)
                    tmp = tmp.scatter(1, indices, topk)
                    logprobs = tmp
            it = torch.distributions.Categorical(logits=logprobs.detach()).sample()  #(batch_size,)
            # print(it.size())
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1)) # gather the logprobs at sampled positions (bs, 1)
            # print(sampleLogprobs.size())
        return it, sampleLogprobs

    def remove_bad_endings(self, sentences):
        new_sentences = []
        for sentence in sentences:
            words = sentence.split(' ')
            while words[-1] in BAD_ENDINGS:
                words = words[:-1]
            new_sentence = ' '.join(words)
            new_sentences.append(new_sentence)
        return new_sentences

    def diverse_beam_search(self,images_features, beam_size,word_map, max_cap_length=50, diversity_prob=0.5): # only support batch_size 1
        '''
        This function only suits for batch_size 1
        :param imgs:
        :param model:
        :param word_map:
        :param beam_size:
        :param max_cap_length:
        :return:
        '''
        self.eval()
        vocab_size = len(word_map)
        num_group = beam_size
        batch_size = images_features.size(0)
        assert batch_size == 1
        rev_word_map = {v: k for k, v in word_map.items()}
        with torch.no_grad():
            complete_seqs = [[] for g in range(num_group)]
            complete_seqs_scores = [[] for g in range(num_group)]
            k_prev_words = [torch.LongTensor([[word_map['<start>']]] *beam_size).cuda() for g in range(num_group)] # (beam_size,)
            top_k_scores = [torch.zeros(beam_size, 1).cuda() for g in range(num_group)] # (beam_size, 1)
            seqs = [torch.LongTensor([[word_map['<start>']]] *beam_size).cuda() for g in range(num_group)]   # (unfinished_num, )
            image_feature_proj = self.relu(self.img_projector(images_features)) # batch_size, 36, hidden_dim
            global_img_feature = torch.mean(image_feature_proj, dim=1)  # (bs, hidden_dim)

            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)   # batch_size, hidden_dim
            image_feature_proj = [image_feature_proj.expand(beam_size, *image_feature_proj.size()[1:]) for g in range(num_group)]  # batch_size, H*W, hidden_dim
            global_img_feature = [global_img_feature.expand(beam_size, global_img_feature.size(-1)) for g in range(num_group)] #  beam_size, hidden_dim,
            h, c = self.init_hidden_state(image_feature_proj[0])
            init_state = (h, c)
            state = [init_state for g in range(num_group)]  #(ht, ct)
            unfinished_num = [beam_size for g in range(num_group)]
            for step in range(max_cap_length):
                previous_idx = []
                for g in range(num_group):
                    if unfinished_num[g] == 0:
                        continue
                    word_embedding = self.embedding(k_prev_words[g]).squeeze(1)  # unfinished_num, embedding_dim
                    xt = torch.cat((word_embedding,  global_img_feature[g]), dim=-1)  # (batch_size, embed_dim + hidden_dim)
                    # print(image_feature_proj[g].size(), xt.size(), state[g])
                    predict_score_t, alpha_t, beta_t, state[g] = self.predict_next_word(image_feature_proj[g], xt, state[g])
                    predict_score_t = torch.log_softmax(predict_score_t, dim=-1)  # (unfinished_num, vocab_size)
                    for i, v in enumerate(previous_idx):
                        predict_score_t[:,int(v)] = predict_score_t[:, int(v)] - diversity_prob
                    top_k_scores_exp = top_k_scores[g].expand((unfinished_num[g], vocab_size))
                    scores = top_k_scores_exp + predict_score_t
                    if step == 0:
                        top_k_scores[g], top_words = scores[0].topk(beam_size, -1, True, True)  # (unfinished_num, beam_size)
                    else:
                        top_k_scores[g], top_words = scores.view(-1).topk(unfinished_num[g], -1, True, True)  # (unfinished_num, beam_size)
                    beam_idx = top_words / vocab_size  # (unfinished_num, )
                    next_word_idx = top_words % vocab_size  # (unfinished_num, )
                    # print('next_word',next_word_idx)
                    seqs[g] = torch.cat([seqs[g][beam_idx], next_word_idx.unsqueeze(1)], dim=1)
                    incomplete_inds = [ind for ind, next_word in enumerate(next_word_idx) if next_word != word_map['<end>']]
                    complete_inds = list(set(range(len(next_word_idx))) - set(incomplete_inds))
                    # Set aside complete sequences
                    if len(complete_inds) > 0:
                        complete_seqs[g].extend(seqs[g][complete_inds].tolist())
                        complete_seqs_scores[g].extend(top_k_scores[g][complete_inds])
                    unfinished_num[g] = unfinished_num[g] - len(complete_inds)  # reduce beam length accordingly
                    if unfinished_num[g] == 0:
                        break
                    # updata sequences
                    seqs[g] = seqs[g][incomplete_inds]
                    #  update state
                    new_state = []
                    for s_idx in range(len(state[g])):
                        new_state.append(state[g][s_idx][beam_idx[incomplete_inds]])
                    state[g] = tuple(new_state)
                    image_feature_proj[g] = image_feature_proj[g][beam_idx[incomplete_inds]]
                    global_img_feature[g] = global_img_feature[g][beam_idx[incomplete_inds]]
                    top_k_scores[g] = top_k_scores[g][incomplete_inds].unsqueeze(1)
                    if g < 2:
                        for i, v in enumerate(k_prev_words[g]):
                            if v.item() not in previous_idx:
                                previous_idx.append(v.item())
                    k_prev_words[g] = next_word_idx[incomplete_inds].unsqueeze(1)

            return_sentences = []
            for g in range(num_group):
                if len(complete_seqs[g]) > 0:
                    i = complete_seqs_scores[g].index(max(complete_seqs_scores[g]))
                    seq = complete_seqs[g][i]
                else:
                    seq = seqs[0][0][:20]
                    seq = [seq[i].item() for i in range(len(seq))]
                sen_idx = [w for w in seq if
                           w not in {word_map['<start>'], word_map['<end>'], word_map['<unk>'], word_map['<pad>']}]
                sentence = ' '.join([rev_word_map[sen_idx[i]] for i in range(len(sen_idx))])
                print(sentence)
                return_sentences.append(sentence)
            return_sentences = self.remove_bad_endings(return_sentences)
            return return_sentences

    def beam_search(self,images_features,  word_map, beam_size=3,max_cap_length=30):
        '''
        This function only suits for batch_size 1
        :param imgs:
        :param model:
        :param word_map:
        :param beam_size:
        :param max_cap_length:
        :return:
        '''
        self.eval()
        assert images_features.size(0) == 1
        batch_size = images_features.size(0)
        rev_word_map = {v: k for k, v in word_map.items()}
        vocab_size = len(word_map)
        complete_seqs =[]
        complete_seqs_scores=[]
        with torch.no_grad():
            k_prev_words = torch.LongTensor([[word_map['<start>']]] *beam_size).cuda() # (beam_size,)
            top_k_scores = torch.zeros(beam_size, 1).cuda() # (beam_size, 1)
            seqs = torch.LongTensor([[word_map['<start>']]] *beam_size).cuda()  # (unfinished_num, )
            image_feature_proj = self.relu(self.img_projector(images_features)) # batch_size, 36, hidden_dim
            global_img_feature = torch.mean(image_feature_proj, dim=1)  # (bs, hidden_dim)
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)   # batch_size, hidden_dim
            # print(global_img_feature.size())
            image_feature_proj = image_feature_proj.expand(beam_size, *image_feature_proj.size()[1:])  # batch_size, H*W, hidden_dim
            global_img_feature = global_img_feature.expand(beam_size, global_img_feature.size(-1)) #  beam_size, hidden_dim,
            h, c = self.init_hidden_state(image_feature_proj)
            state = (h, c)
            unfinished_num = beam_size
            for step in range(max_cap_length):
                word_embedding = self.embedding(k_prev_words).squeeze(1) # unfinished_num, embedding_dim
                xt = torch.cat((word_embedding, global_img_feature), dim=-1) # (batch_size, 2*embed_dim + hidden_dim)
                predict_score_t, alpha_t, beta_t, state = self.predict_next_word(image_feature_proj, xt, state)
                predict_score_t = torch.log_softmax(predict_score_t,dim=-1) #(unfinished_num, vocab_size)
                top_k_scores_exp = top_k_scores.expand((unfinished_num, vocab_size))
                scores = top_k_scores_exp + predict_score_t
                if step == 0:
                    top_k_scores, top_words = scores[0].topk(beam_size, -1, True, True)  # (unfinished_num, beam_size)
                else:
                    top_k_scores, top_words = scores.view(-1).topk(unfinished_num, -1, True, True) # (unfinished_num, beam_size)
                beam_idx = top_words // vocab_size  # (unfinished_num, )
                next_word_idx = top_words % vocab_size  # (unfinished_num, )
                # print('next_word',next_word_idx)
                seqs = torch.cat([seqs[beam_idx], next_word_idx.unsqueeze(1)], dim=1)
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_idx) if next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_idx))) - set(incomplete_inds))
                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                unfinished_num = unfinished_num - len(complete_inds)  # reduce beam length accordingly
                if unfinished_num == 0:
                    break
                # updata sequences
                seqs = seqs[incomplete_inds]
                #  update state
                new_state = []
                for s_idx in range(len(state)):
                    new_state.append(state[s_idx][beam_idx[incomplete_inds]])
                state = tuple(new_state)
                image_feature_proj = image_feature_proj[beam_idx[incomplete_inds]]
                global_img_feature = global_img_feature[beam_idx[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_idx[incomplete_inds].unsqueeze(1)
                # Break if things have been going on too long
            if len(complete_seqs) > 0:
                i = complete_seqs_scores.index(max(complete_seqs_scores))
                seq = complete_seqs[i]
            else:
                seq = seqs[0][:20]
                seq = [seq[i].item() for i in range(len(seq))]
            sen_idx = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<unk>'], word_map['<pad>']}]
            sentence = [' '.join([rev_word_map[sen_idx[i]] for i in range(len(sen_idx))])]
            sentence = self.remove_bad_endings(sentence)
            return sentence, sen_idx

    def greedy_search(self,images_features,  word_map, max_cap_length=20):
        self.eval()
        batch_size = images_features.size(0)
        rev_word_map = {v: k for k, v in word_map.items()}
        complete_seq =[]
        with torch.no_grad():
            k_prev_words = torch.zeros(batch_size, max_cap_length).long().cuda() # (batch_size, caption_length)
            k_prev_words[:, 0] = word_map['<start>'] # the first word is '<start>'
            seqs_temp = [[word_map['<start>']] for _ in range(batch_size)]
            image_feature_proj = self.relu(self.img_projector(images_features)) # batch_size, hidden_dim, H, W
            global_img_feature = torch.mean(image_feature_proj, dim=1)  # (bs, hidden_dim)
            h, c = self.init_hidden_state(image_feature_proj)
            state = (h, c)
            for step in range(max_cap_length-1):
                word_embedding = self.embedding(k_prev_words[:, step]) # batch_size, embedding_dim
                if global_img_feature.dim() == 1:
                    global_img_feature = global_img_feature.unsqueeze(0)
                xt = torch.cat((word_embedding,  global_img_feature), dim=-1) # (batch_size, embed_dim + hidden_dim)
                predict_score_t, alpha_t, beta_t, state = self.predict_next_word(image_feature_proj, xt, state)
                predict_score_t = torch.log_softmax(predict_score_t,dim=-1) #(batch_size, vocab_size)
                top_scores, top_words = predict_score_t.topk(1, -1, True, True)
                if step == 0:
                    finished = top_words == word_map['<end>']
                    unfinished = ~finished
                else:
                    current_finished = top_words == word_map['<end>']
                    unfinished = unfinished * ~current_finished
                top_words = top_words * unfinished.type_as(top_words)
                k_prev_words[:, step+1] = top_words[:,0]
                for bs in range(batch_size):
                    seqs_temp[bs].append(int(top_words[bs][0].cpu().numpy()))
            for bs in range(batch_size):
                sen = seqs_temp[bs]
                sen_idx = [w for w in sen if w not in {word_map['<start>'], word_map['<end>'], word_map['<unk>'],word_map['<pad>']}]
                # print(sen_idx)
                sentence = ' '.join([rev_word_map[sen_idx[i]] for i in range(len(sen_idx))])
                # print(sentence)
                complete_seq.append(sentence)
            complete_seq = self.remove_bad_endings(complete_seq)
            return complete_seq, sen_idx

    def lrp_linear_eps(self, r_out, forward_input, forward_output, weight):
        '''

        :param r_out:  relevance of the output (out_feature,)
        :param forward_input: the output tensor (in_feature, )
        :param forward_output: the input tensor (out_feature, )
        :param weight:  weight tensor shape (out_feature, in_feature)
        :return: r_in (in_feature,)
        '''
        assert r_out.dim() == 1
        assert forward_input.dim() == 1
        assert weight.dim() == 2
        attribution = weight * forward_input #(out_feature, in_feature)
        if type(forward_output) == bool:
            forward_output = torch.matmul(forward_input, weight.transpose(0,1))
            # print('matml', forward_output.size())
        forward_output_eps = self.EPS * forward_output.sign() + forward_output  # Z.sign() returns -1 or 0 or 1

        forward_output_eps.masked_fill_(forward_output_eps == 0, self.EPS)  #(out_feature,)
        # print(forward_output_eps.size())
        attribution_norm = attribution.transpose(0,1) / forward_output_eps #(in_feature, out_feature)
        # print(attribution_norm.size())
        relevance_input = torch.sum(attribution_norm * r_out, dim=-1) #(in_feature,)
        assert relevance_input.size() == forward_input.size()
        torch.cuda.empty_cache()
        return relevance_input

    def lrp_mha(self, alpha, value, r_context, context):
        '''

        :param alpha:  shape is num_head, num_pixel
        :param value: shape is  num_pixel hiddendim
        :param r_context: shape is  1, hiddendim
        :param context shape is 1, hiddendim
        :return:
        '''
        num_head = alpha.size(0)
        num_pixel = alpha.size(1)
        num_query = r_context.size(0)  #(should be 1 if we use ht ad the query)
        '''spread single head'''
        d_k = self.model.hidden_dim//num_head #(should be 64 if using vgg and 8 heads)
        r_context = r_context.clone().contiguous().view(num_query, num_head, d_k).transpose(0, 1) #(num_head, num_query, d_k)
        context = context.clone().contiguous().view(num_query, num_head, d_k).transpose(0,1)  #(num_head, num_query, d_k)
        value = value.clone().contiguous().view(num_pixel, num_head, d_k).transpose(0, 1) #(num_head, num_pixel, d_k)
        r_value = value.clone()
        for h in range(num_head):
            for i in range(num_pixel):
                r = self.lrp_linear_eps(r_out=r_context[h,0],
                                                forward_input=value[h,i]*alpha[h,i],
                                                forward_output=context[h,0],
                                                weight=torch.eye(d_k).cuda())
                r_value[h, i] = r
        # print(r_value[:, 0].squeeze())
        r_value = r_value.transpose(0, 1).contiguous().view(num_pixel, self.model.hidden_dim)
        # print(r_value[0])
        return r_value

    def get_lrp_weight_step(self, predictions_t, rev_word_map, ht_, context_aoa):
        batch_size, vocab_size = predictions_t.size()
        with torch.no_grad():
            weight_of_context_aoa = torch.zeros(batch_size, self.hidden_dim).cuda()
            weight_of_ht = torch.zeros(batch_size, self.hidden_dim).cuda()
            for b in range(batch_size):
                predicted_labels = torch.argmax(predictions_t[b], dim=-1)  # (the predicted label of image b)  (max_length)
                word_t = predicted_labels.item()
                if rev_word_map[word_t] in STOP_WORDS + ['<start>','<end>','<pad>','<unk>']:
                    continue
                else:
                    word_relevance = torch.zeros(self.vocab_size).cuda()
                    word_relevance[word_t] = predictions_t[b][word_t]
                    r_h2t_context_aoa = self.lrp_linear_eps(r_out=word_relevance,
                                                            forward_input=ht_[b] + context_aoa[b],
                                                            forward_output=predictions_t[b],
                                                            weight=self.fc.weight)
                    r_h2t = self.lrp_linear_eps(r_out=r_h2t_context_aoa,
                                                forward_input=ht_[b],
                                                forward_output=ht_[b] + context_aoa[b],
                                                weight=torch.eye(self.hidden_dim).cuda())
                    weight_of_ht[b] = r_h2t
                    r_context_aoa = self.lrp_linear_eps(r_out=r_h2t_context_aoa,
                                                        forward_input=context_aoa[b],
                                                        forward_output=ht_[b] + context_aoa[b],
                                                        weight=torch.eye(self.hidden_dim).cuda())
                    weight_of_context_aoa[b] = r_context_aoa
            weight_of_context_aoa = LRPutil.normalize_relevance(weight_of_context_aoa, dim=-1)
            weight_of_ht = LRPutil.normalize_relevance(weight_of_ht, dim=-1)
            return weight_of_context_aoa, weight_of_ht

    def forwardlrp_context(self, images_features, encoded_captions, caption_lengths, rev_word_map):
        def lstm_forward(x, h, c, wi, wh, bi, bh):
            z = torch.matmul(x, wi.transpose(0, 1))  # (batch_size, 4*hidden_size)
            z = z + torch.matmul(h, wh.transpose(0, 1))  # (batch_size, 4*hidden_size)
            z = z + bi + bh
            z0, z1, z2, z3 = z.chunk(4, dim=1)
            i = torch.sigmoid(z0)
            f = torch.sigmoid(z1)
            c = f * c + i * torch.tanh(z2)
            o = torch.sigmoid(z3)
            h = o * torch.tanh(c)
            c = c
            return h, c, z2, i, f  # (batch_size, 512)
        batch_size = images_features.size(0)
        # print(image_features.size(), avg_feature.size())
        image_feature_proj_before_act = self.img_projector(images_features)
        image_feature_proj = self.relu(image_feature_proj_before_act)  # (bs, 36, hiddendim)
        global_img_feature = torch.mean(image_feature_proj, dim=1)  # (bs, hidden_dim)
        h, c = self.init_hidden_state(image_feature_proj)
        state = (h, c)

        key = self.decoder_k_proj(image_feature_proj)
        value = self.decoder_v_proj(image_feature_proj)
        max_length = max(caption_lengths) - 1
        predictions = torch.zeros(batch_size, max_length, self.vocab_size).cuda()
        weighted_predictions = torch.zeros(batch_size, max_length, self.vocab_size).cuda()
        for t in range(max_length):
            word_embedding = self.embedding(encoded_captions[:, t])
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            xt_ = torch.cat((word_embedding, global_img_feature), dim=-1)  # (batch_size, hidden_dim + embed_dim)
            h_, c_, g_, i_act_, f_act_ = lstm_forward(xt_, state[0], state[1], self.LanguageLSTM.weight_ih,
                                                      self.LanguageLSTM.weight_hh,
                                                      self.LanguageLSTM.bias_ih, self.LanguageLSTM.bias_hh)
            context_, alpha_t_ = self.decoder_multihead_attention(h_, key, value)  # (bs, hidden_dim) alpha: (bs, num_pixel)
            context_aoa_gate_ = self.decoder_aoa_linear_gate(h_)
            context_aoa_linear_ = self.decoder_aoa_linear(context_)
            context_aoa_ = torch.sigmoid(context_aoa_gate_) * context_aoa_linear_  # (bs, hiddendim)
            predict_score_t = self.fc(self.dropout(context_aoa_ + h_))  # (bs, vocab_size)
            state = (h_, c_)
            predictions[:, t, :] = predict_score_t
            weight_context_aoa, weight_ht = self.get_lrp_weight_step(predict_score_t, rev_word_map, h_, context_aoa_)
            weighted_prediction_t = self.fc(self.dropout(weight_context_aoa*context_aoa_ + h_*weight_ht))
            weighted_predictions[:, t,:] = weighted_prediction_t
        return predictions, weighted_predictions, max_length

    def sample_lrp(self, images_features, rev_word_map, word_map, caption_lengths, opt={}):
        def lstm_forward(x, h, c, wi, wh, bi, bh):
            z = torch.matmul(x, wi.transpose(0, 1))  # (batch_size, 4*hidden_size)
            z = z + torch.matmul(h, wh.transpose(0, 1))  # (batch_size, 4*hidden_size)
            z = z + bi + bh
            z0, z1, z2, z3 = z.chunk(4, dim=1)
            i = torch.sigmoid(z0)
            f = torch.sigmoid(z1)
            c = f * c + i * torch.tanh(z2)
            o = torch.sigmoid(z3)
            h = o * torch.tanh(c)
            c = c
            return h, c, z2, i, f  # (batch_size, 512)
        batch_size = images_features.size(0)
        sample_method = opt.get('sample_method', 'greedy')
        temperature = opt.get('temperature', 1.0)
        max_length = max(caption_lengths) - 1
        image_feature_proj = self.relu(self.img_projector(images_features))  # (bs, hiddendim, H, W)
        global_img_feature = torch.mean(image_feature_proj, dim=1)  # (bs, hidden_dim)
        key = self.decoder_k_proj(image_feature_proj)
        value = self.decoder_v_proj(image_feature_proj)
        h, c = self.init_hidden_state(image_feature_proj)
        state = (h, c)
        seq = torch.zeros(batch_size,max_length).long().cuda()
        seq_logprobs = torch.zeros(batch_size, max_length).cuda()
        for t in range(max_length):
            if t == 0:
                it = torch.ones(batch_size).long().cuda() * word_map['<start>']
            word_embedding = self.embedding(it)
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            xt_ = torch.cat((word_embedding, global_img_feature), dim=-1)  # (batch_size, hidden_dim + embed_dim)
            h_, c_, g_, i_act_, f_act_ = lstm_forward(xt_, state[0], state[1], self.LanguageLSTM.weight_ih,
                                                      self.LanguageLSTM.weight_hh,
                                                      self.LanguageLSTM.bias_ih, self.LanguageLSTM.bias_hh)
            context_, alpha_t_ = self.decoder_multihead_attention(h_, key,
                                                                  value)  # (bs, hidden_dim) alpha: (bs, num_pixel)
            context_aoa_gate_ = self.decoder_aoa_linear_gate(h_)
            context_aoa_linear_ = self.decoder_aoa_linear(context_)
            context_aoa_ = torch.sigmoid(context_aoa_gate_) * context_aoa_linear_  # (bs, hiddendim)
            predict_score_t = self.fc(self.dropout(context_aoa_ + h_))  # (bs, vocab_size)
            state = (h_, c_)
            predict_score_t = torch.log_softmax(predict_score_t,dim=-1)

            weight_context_aoa, weight_ht = self.get_lrp_weight_step(predict_score_t, rev_word_map, h_, context_aoa_)
            weight_prediction_t = self.fc(context_aoa_ * weight_context_aoa + weight_ht * h_)
            predict_score_t = torch.log_softmax(weight_prediction_t, dim=-1)

            it, sampleLpgprobs = self.sample_next_word(predict_score_t, sample_method, temperature)
            # sample the next word
            if t == 0:
                current_finished = it == word_map['<end>']
                unfinished = ~current_finished
            else:
                current_finished = it == word_map['<end>']
                unfinished = unfinished * ~current_finished
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            seq_logprobs[:, t] = sampleLpgprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break
        return seq, seq_logprobs, max_length



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import config
    import json
    import glob
    img_filepath = '/home/sunjiamei/work/ImageCaptioning/dataset/coco/images/val2017/000000015746.jpg'
    # img_filepath = '/home/sunjiamei/work/ImageCaptioning/dataset/flickr30k/Flickr30k_Dataset/1009434119.jpg'
    parser = config.imgcap_aoa_argument_parser()
    args = parser.parse_args()
    args.dataset = 'coco2017'
    args.weight = glob.glob('../output/aoa/vgg16/coco2017/BEST_checkpoint_coco2017_epoch34*')[0]
    # args.dataset = 'flickr30k'
    # args.weight = glob.glob('../output/aoa/vgg16/flickr30k/BEST_checkpoint_flickr30k_epoch31*')[0]
    word_map_path = f'../dataset/wordmap_{args.dataset}.json'
    word_map = json.load(open(word_map_path, 'r'))
    #
    for explainer in [ExplainAOAAttention(args, word_map),ExplainAOAGradient(args, word_map),ExplainAOAGuidedGradient(args, word_map),
                      ExplainAOAGradCam(args, word_map),ExplainAOAGuidedGradCam(args, word_map)]:
        for i in range(8):
            explainer.explain_caption(img_filepath, head_idx=i, t_list=[3,7])