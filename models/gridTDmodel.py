import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import models.resnet as resnet
import models.vgg as vgg
import numpy as np
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
BAD_ENDINGS = ['with','in','on','of','a','at','to','for','an','this','his','her','that','the', 'and']

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


class AdaptiveLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AdaptiveLSTMCell, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.x_gate = nn.Linear(input_size, hidden_size)
        self.h_gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, inp, states):
        h_old, c_old = states
        ht, ct = self.lstm_cell(inp, (h_old, c_old))
        sen_gate = torch.sigmoid(self.x_gate(inp) + self.h_gate(h_old))
        st = sen_gate * torch.tanh(ct)
        return ht, ct, st


class AdaptiveAttention(nn.Module):
    def __init__(self, hidden_dim, n_pixel):
        super(AdaptiveAttention, self).__init__()
        self.hidden_dim = hidden_dim  # the rnn size
        self.num_pixel = n_pixel
        self.W_v_proj = nn.Linear(hidden_dim,n_pixel)
        self.W_s_proj = nn.Linear(hidden_dim,n_pixel)
        self.W_g_proj = nn.Linear(hidden_dim,n_pixel,bias=False)
        self.w_h = nn.Linear(n_pixel, 1, bias=False)

    def forward(self, V, ht, st):
        """
        V: the spatial image of size (batch_size,hidden_size,num_pixel)
        decoder_out: the decoder hidden state of shape (batch_size, hidden_size)
        st: visual sentinal returned by the Sentinal class, of shape: (batch_size, hidden_size)
        """
        V = V.transpose(1,2)  #(bs, num_pixel,hidden_dim)
        batch_size = ht.size(0)
        img_proj = self.W_v_proj(V) # (bs, num_pixel, num_pixel)
        # print('img_proj', img_proj.size())
        ht_proj = self.W_g_proj(ht)        # (-1, num_pixel)
        # print('ht_proj', ht_proj.size())
        one_matrix = torch.ones(batch_size,1, self.num_pixel).cuda() # (bs, 1, num_pixel)
        # print('one_matrix', one_matrix.size())
        ht_proj_expand = torch.bmm(ht_proj.unsqueeze(2), one_matrix) #(bs, num_pixel, num_pixel)
        # print('ht_proj_expand', ht_proj_expand.size())
        z_t = self.w_h(torch.tanh(img_proj+ht_proj_expand)) #(bs, num_pixel, 1)
        # print('zt', z_t.size())
        alpha_t = torch.softmax(z_t, dim=1) #(bs, num_pixel,1)
        # print('alpha', alpha_t.size())
        context_t = torch.sum(V * alpha_t, dim=1) #(bs, hidden_dim,)
        # print('ct', c_t.size())
        attention_vs = self.w_h(torch.tanh(self.W_s_proj(st)+ht_proj)) #(bs, 1)
        # print('attention_s', attention_vs.size())
        concatenates = torch.cat([z_t, attention_vs.unsqueeze(-1)], dim=1) #(bs, num_pixel+1,1)
        # print('concatenates', concatenates.size())
        alpha_t_hat = torch.softmax(concatenates, dim=1) #(bs, num_pixel+1, 1)
        # print('alpha_t_hat', alpha_t_hat.size())
        beta_t = alpha_t_hat[:,-1]  #(bs, 1)
        # print(beta_t.size())
        alpha_t = alpha_t.squeeze(2)
        c_t_hat = beta_t * st + (1-beta_t) * context_t
        return c_t_hat, context_t, alpha_t, beta_t


class GridTDModel(nn.Module):
    '''
    This model add another lstm layer on top of the adaptive attention model
    '''
    EPS = LRPutil.EPSILON
    def __init__(self, embed_dim, hidden_dim, vocab_size, encoder_type):
        super(GridTDModel, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.encoder_type = encoder_type
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(0.5)
        # the image encoder to generate image features (bs, C, H, W)
        self.img_encoder = Encoder(self.encoder_type)
        self.encoder_raw_dim = self.img_encoder.feat_dim
        print(f'==========Encoded image feature dim is {self.encoder_raw_dim}==========')
        self.img_projector = nn.Conv2d(self.encoder_raw_dim, self.hidden_dim, kernel_size=1,stride=1)
        self.global_img_feature_proj = nn.Linear(self.encoder_raw_dim, self.embed_dim)
        self.LanguageLSTM = nn.LSTMCell(2*hidden_dim, hidden_dim)
        self.AdaLSTM = AdaptiveLSTMCell(embed_dim*2 + hidden_dim, hidden_dim)
        self.AdaAttention = AdaptiveAttention(self.hidden_dim, 196)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()

    def init_hidden_state(self, V):
        h = torch.zeros(V.shape[0], self.hidden_dim).cuda()
        c = torch.zeros(V.shape[0], self.hidden_dim).cuda()
        return h, c

    def predict_next_word(self,image_feature_proj, xt, states):
        h1t, c1t, h2t, c2t = states  # (bs, hidden_dim, )
        h1t, c1t, st = self.AdaLSTM(xt, (h1t, c1t))  #(bs, hidden_dim, )
        context_t_hat, context_t, alpha_t, beta_t = self.AdaAttention(image_feature_proj, h1t, st)  #(bs, hidden_dim) alpha: (bs, num_pixel), beta:(bs, 1)
        language_input = torch.cat((context_t_hat, h1t), dim=-1) #(bs, 2*hiddendim)
        h2t, c2t = self.LanguageLSTM(language_input, (h2t, c2t))  #(bs, hiddendim)
        predict_score_t = self.fc(self.dropout(context_t_hat + h2t))  # (bs, vocab_size)
        return predict_score_t, alpha_t, beta_t, (h1t, c1t, h2t, c2t)

    def forward(self, images, encoded_captions, caption_lengths, ss_prob):
        """
        images: the encoded images from the encoder, of shape (batch_size, C, H, W)
        global_features: the global image features returned by the Encoder, of shape: (batch_size, hidden_dim)
        encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        """
        batch_size = images.size(0)
        image_features, avg_feature = self.img_encoder(images) # (bs, fea_dim, H, W), (bs, fea_dim)
        num_pixels = image_features.size(-1) * image_features.size(-2)
        # print(image_features.size(), avg_feature.size())
        image_feature_proj = self.relu(self.img_projector(image_features)) # (bs, hiddendim, H, W)
        # print(image_feature_proj.size())
        global_img_feature = self.relu(self.global_img_feature_proj(avg_feature)) #(bs, embedding_dim)
        # print(global_img_feature.size())
        image_feature_proj = image_feature_proj.contiguous()
        image_feature_proj = image_feature_proj.view(batch_size, self.hidden_dim, -1) # (bs, hidden_dim, num_pixel)
        # print(image_feature_proj.size())
        h1, c1 = self.init_hidden_state(image_feature_proj)
        h2, c2 = self.init_hidden_state(image_feature_proj)
        state = (h1, c1, h2, c2)
        max_length = max(caption_lengths)-1
        predictions = torch.zeros(batch_size, max_length, self.vocab_size).cuda()
        alphas = torch.zeros(batch_size, max_length , num_pixels).cuda()
        betas = torch.zeros(batch_size, max_length,1).cuda()
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
                word_embedding = self.embedding(encoded_captions[:,t])
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            xt = torch.cat((state[2], global_img_feature, word_embedding), dim=-1)   # (batch_size, 2*embed_dim + hidden_dim)
            predict_score_t, alpha_t, beta_t, state = self.predict_next_word(image_feature_proj, xt, state)
            predictions[:, t,:] = predict_score_t
            alphas[:, t, :] = alpha_t
            betas[:, t, :] = beta_t
            last_scores = torch.log_softmax(predict_score_t,-1)
            # print(last_scores.size())
            last_label = torch.argmax(last_scores, -1)  #(batch_size, )
            # print(last_label.size())
        return predictions, alphas, betas, last_scores, max_length

    def sample(self, images, word_map, caption_lengths, opt={}):

        batch_size = images.size(0)
        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        max_length = max(caption_lengths) - 1
        image_features, avg_feature = self.img_encoder(images)  # (bs, fea_dim, H, W), (bs, fea_dim)

        # print(image_features.size(), avg_feature.size())
        image_feature_proj = self.relu(self.img_projector(image_features))  # (bs, hiddendim, H, W)
        # print(image_feature_proj.size())
        global_img_feature = self.relu(self.global_img_feature_proj(avg_feature))  # (bs, embedding_dim)
        # print(global_img_feature.size())
        image_feature_proj = image_feature_proj.contiguous()
        image_feature_proj = image_feature_proj.view(batch_size, self.hidden_dim, -1)  # (bs, hidden_dim, num_pixel)
        state = self.init_hidden_state(image_feature_proj) + self.init_hidden_state(image_feature_proj)
        seq = torch.zeros(batch_size,max_length).long().cuda()
        seq_logprobs = torch.zeros(batch_size, max_length).cuda()
        for t in range(max_length):
            if t == 0:
                it = torch.ones(batch_size).long().cuda() * word_map['<start>']
            word_embedding = self.embedding(it)
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            xt = torch.cat((state[2], global_img_feature, word_embedding), dim=-1)  # (batch_size, 2*embed_dim)
            predict_score_t, alpha_t, beta_t, state = self.predict_next_word(image_feature_proj, xt, state)
            predict_score_t = torch.log_softmax(predict_score_t,dim=-1)
            it, sampleLpgprobs = self.sample_next_word(predict_score_t, sample_method, temperature)
            # sample the next word
            if t == 0:
                finished = it == word_map['<end>']
                unfinished = ~finished
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
            bad_sentence= False
            words = sentence.split(' ')
            if len(words) == 0:
                bad_sentence = True
            else:
                while words[-1] in BAD_ENDINGS:
                    words = words[:-1]
                    if len(words) == 0:
                        bad_sentence = True
                        break
            if bad_sentence:
                new_sentences.append(sentence)
            else:
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
            image_feature_proj = image_feature_proj.view(batch_size, image_feature_proj.size(1), -1) # batch_size, hidden_dim, H*W
            global_img_feature = self.relu(self.global_img_feature_proj(avg_feature))
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)   # batch_size, hidden_dim
            image_feature_proj = [image_feature_proj.expand(beam_size, *image_feature_proj.size()[1:]) for g in range(num_group)] # beam_size, hidden_dim, H*W
            global_img_feature = [global_img_feature.expand(beam_size, global_img_feature.size(-1)) for g in range(num_group)] #  beam_size, hidden_dim,
            init_state = self.init_hidden_state(image_feature_proj[0]) + self.init_hidden_state(image_feature_proj[0])
            state = [init_state for g in range(num_group)]  #(ht, ct)
            unfinished_num = [beam_size for g in range(num_group)]
            for step in range(max_cap_length):
                previous_idx = []
                for g in range(num_group):
                    if unfinished_num[g] == 0:
                        continue
                    word_embedding = self.embedding(k_prev_words[g]).squeeze(1)  # unfinished_num, embedding_dim
                    xt = torch.cat((state[g][2], global_img_feature[g], word_embedding), dim=-1)  # (batch_size, 2*embed_dim + hidden_dim)
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
                    beam_idx = top_words // vocab_size  # (unfinished_num, )
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

    def beam_search(self,imgs,  word_map, beam_size=3,max_cap_length=20):
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
            # print(image_feature_proj.size())
            image_feature_proj = image_feature_proj.view(batch_size, image_feature_proj.size(1), -1) # batch_size, hidden_dim, H*W
            global_img_feature = self.relu(self.global_img_feature_proj(avg_feature))
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)   # batch_size, hidden_dim
            # print(global_img_feature.size())
            image_feature_proj = image_feature_proj.expand(beam_size, *image_feature_proj.size()[1:]) # beam_size, hidden_dim, H*W
            global_img_feature = global_img_feature.expand(beam_size, global_img_feature.size(-1)) #  beam_size, hidden_dim,
            state = self.init_hidden_state(image_feature_proj) +  self.init_hidden_state(image_feature_proj)  #(ht, ct)
            unfinished_num = beam_size
            for step in range(max_cap_length):
                word_embedding = self.embedding(k_prev_words).squeeze(1) # unfinished_num, embedding_dim
                xt = torch.cat((state[2], global_img_feature, word_embedding), dim=-1) # (batch_size, 2*embed_dim + hidden_dim)
                predict_score_t, alpha_t, beta_t, state = self.predict_next_word(image_feature_proj, xt, state)
                predict_score_t = torch.log_softmax(predict_score_t,dim=-1) #(unfinished_num, vocab_size)
                top_k_scores_exp = top_k_scores.expand((unfinished_num, vocab_size))
                scores = top_k_scores_exp + predict_score_t
                if step == 0:
                    top_k_scores, top_words = scores[0].topk(beam_size, -1, True, True)  # (unfinished_num, beam_size)
                else:
                    top_k_scores, top_words = scores.view(-1).topk(unfinished_num, -1, True, True) # (unfinished_num, beam_size)
                beam_idx = top_words / vocab_size  # (unfinished_num, )
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
            sen_idx = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<unk>'],word_map['<pad>']}]
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
            image_feature_proj = image_feature_proj.view(batch_size, image_feature_proj.size(1), -1)
            global_img_feature = self.relu(self.global_img_feature_proj(avg_feature))  # batch_size, hidden_dim
            state = self.init_hidden_state(image_feature_proj) +  self.init_hidden_state(image_feature_proj)#(ht, ct)
            for step in range(max_cap_length-1):
                word_embedding = self.embedding(k_prev_words[:, step]) # batch_size, embedding_dim
                if global_img_feature.dim() == 1:
                    global_img_feature = global_img_feature.unsqueeze(0)
                xt = torch.cat((state[2], global_img_feature, word_embedding), dim=-1) # (batch_size, 2*embed_dim + hidden_dim)
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
            return complete_seq, seqs_temp

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

    def get_lrp_weight_step(self, predictions_t, rev_word_map, h2t_, context_hat):
        batch_size, vocab_size = predictions_t.size()
        with torch.no_grad():
            weight_of_context_hat = torch.zeros(batch_size, self.hidden_dim).cuda()
            weight_of_h2t = torch.zeros(batch_size, self.hidden_dim).cuda()
            for b in range(batch_size):
                predicted_labels = torch.argmax(predictions_t[b], dim=-1)  # (the predicted label of image b)  (max_length)
                word_t = predicted_labels.item()
                if rev_word_map[word_t] in STOP_WORDS + ['<start>','<end>','<pad>','<unk>']:
                    continue
                else:
                    word_relevance = torch.zeros(self.vocab_size).cuda()
                    word_relevance[word_t] = predictions_t[b][word_t]
                    r_h2t_context_hat = self.lrp_linear_eps(r_out=word_relevance,
                                                            forward_input=h2t_[b] + context_hat[b],
                                                            forward_output=predictions_t[b],
                                                            weight=self.fc.weight)
                    r_h2t = self.lrp_linear_eps(r_out=r_h2t_context_hat,
                                                      forward_input=h2t_[b],
                                                      forward_output=h2t_[b] + context_hat[b],
                                                      weight=torch.eye(self.hidden_dim).cuda())
                    weight_of_h2t[b] = r_h2t
                    r_context_hat = self.lrp_linear_eps(r_out=r_h2t_context_hat,
                                                        forward_input=context_hat[b],
                                                        forward_output=h2t_[b] + context_hat[b],
                                                        weight=torch.eye(self.hidden_dim).cuda())
                    weight_of_context_hat[b] = r_context_hat
            weight_of_context_hat = LRPutil.normalize_relevance(weight_of_context_hat,dim=-1)
            weight_of_h2t = LRPutil.normalize_relevance(weight_of_h2t, dim=-1)
            return weight_of_context_hat, weight_of_h2t

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
        h1, c1 = self.init_hidden_state(image_feature_proj)
        h2, c2 = self.init_hidden_state(image_feature_proj)
        state = (h1, c1, h2, c2)
        global_img_feature_before_act = self.global_img_feature_proj(avg_feature)  # (bs, hidden_dim)
        global_img_feature = self.relu(global_img_feature_before_act)
        max_length = max(caption_lengths) - 1
        predictions = torch.zeros(batch_size, max_length, self.vocab_size).cuda()
        weighted_predictions = torch.zeros(batch_size, max_length, self.vocab_size).cuda()
        for t in range(max_length):
            word_embedding = self.embedding(encoded_captions[:, t])
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            x1t_ = torch.cat((state[2], global_img_feature, word_embedding), dim=-1)  # (batch_size, hidden_dim + embed_dim)
            h1_, c1_, g1_, i1_act_, f1_act_ = lstm_forward(x1t_, state[0], state[1], self.AdaLSTM.lstm_cell.weight_ih,
                                                          self.AdaLSTM.lstm_cell.weight_hh,
                                                          self.AdaLSTM.lstm_cell.bias_ih, self.AdaLSTM.lstm_cell.bias_hh)
            sen_gate = torch.sigmoid(self.AdaLSTM.x_gate(x1t_) + self.AdaLSTM.h_gate(h1_))
            st_ = sen_gate * torch.tanh(c1_)
            context_t_hat_, context_t_, alpha_t_, beta_t_ = self.AdaAttention(image_feature_proj, h1_, st_)  # (1, hidden_dim), (1, num_pixel), (1,1)
            x2t_ = torch.cat((context_t_hat_, h1_), dim=-1)
            h2_, c2_, g2_, i2_act_, f2_act_ = lstm_forward(x2t_, state[2], state[3], self.LanguageLSTM.weight_ih,
                                                          self.LanguageLSTM.weight_hh,
                                                          self.LanguageLSTM.bias_ih, self.LanguageLSTM.bias_hh)
            predict_score_t = self.fc(context_t_hat_ + h2_)  # (bs, vocab_size)
            state = (h1_, c1_, h2_, c2_)
            # print(predict_score_t.size(), x1t_.size(), h1_.size(), c1_.size(), g1_.size(), i1_act_.size(), f1_act_.size(),
            #       alpha_t_.size(), x2t_.size(), h2_.size(), c2_.size(), g2_.size(), i2_act_.size(), f2_act_.size(), context_t_.size(),
            #       context_t_hat_.size(), st_.size(), beta_t_.size())
            predictions[:, t, :] = predict_score_t
            weight_context_hat, weight_h2t = self.get_lrp_weight_step(predict_score_t, rev_word_map, h2_, context_t_hat_)
            weight_prediction_t = self.fc(context_t_hat_*weight_context_hat + weight_h2t * h2_)
            weighted_predictions[:, t, :] = weight_prediction_t
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
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        max_length = max(caption_lengths) - 1
        image_features, avg_feature = self.img_encoder(images)  # (bs, fea_dim, H, W), (bs, fea_dim)

        # print(image_features.size(), avg_feature.size())
        image_feature_proj = self.relu(self.img_projector(image_features))  # (bs, hiddendim, H, W)
        # print(image_feature_proj.size())
        global_img_feature = self.relu(self.global_img_feature_proj(avg_feature))  # (bs, embedding_dim)
        # print(global_img_feature.size())
        image_feature_proj = image_feature_proj.contiguous()
        image_feature_proj = image_feature_proj.view(batch_size, self.hidden_dim, -1)  # (bs, hidden_dim, num_pixel)
        state = self.init_hidden_state(image_feature_proj) + self.init_hidden_state(image_feature_proj)
        seq = torch.zeros(batch_size,max_length).long().cuda()
        seq_logprobs = torch.zeros(batch_size, max_length).cuda()

        for t in range(max_length):
            if t == 0:
                it = torch.ones(batch_size).long().cuda() * word_map['<start>']
            word_embedding = self.embedding(it)
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            x1t_ = torch.cat((state[2], global_img_feature, word_embedding), dim=-1)  # (batch_size, hidden_dim + embed_dim)
            h1_, c1_, g1_, i1_act_, f1_act_ = lstm_forward(x1t_, state[0], state[1], self.AdaLSTM.lstm_cell.weight_ih,
                                                          self.AdaLSTM.lstm_cell.weight_hh,
                                                          self.AdaLSTM.lstm_cell.bias_ih, self.AdaLSTM.lstm_cell.bias_hh)
            sen_gate = torch.sigmoid(self.AdaLSTM.x_gate(x1t_) + self.AdaLSTM.h_gate(h1_))
            st_ = sen_gate * torch.tanh(c1_)
            context_t_hat_, context_t_, alpha_t_, beta_t_ = self.AdaAttention(image_feature_proj, h1_, st_)  # (1, hidden_dim), (1, num_pixel), (1,1)
            x2t_ = torch.cat((context_t_hat_, h1_), dim=-1)
            h2_, c2_, g2_, i2_act_, f2_act_ = lstm_forward(x2t_, state[2], state[3], self.LanguageLSTM.weight_ih,
                                                          self.LanguageLSTM.weight_hh,
                                                          self.LanguageLSTM.bias_ih, self.LanguageLSTM.bias_hh)
            state = (h1_, c1_, h2_, c2_)
            predict_score_t = self.fc(context_t_hat_ + h2_)  # (bs, vocab_size)
            weight_context_hat, weight_h2t = self.get_lrp_weight_step(predict_score_t, rev_word_map, h2_, context_t_hat_)
            weight_prediction_t = self.fc(context_t_hat_ * weight_context_hat + weight_h2t * h2_)
            predict_score_t = torch.log_softmax(weight_prediction_t,dim=-1)
            it, sampleLpgprobs = self.sample_next_word(predict_score_t, sample_method, temperature)
            # sample the next word
            if t == 0:
                finished = it == word_map['<end>']
                unfinished = ~finished
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
class ExplainGridTDAttention(object):
    EPS = LRPutil.EPSILON
    EX_TYPE = 'lrp'
    def __init__(self, args, word_map, model=None):
        super(ExplainGridTDAttention, self).__init__()
        self.args = args
        self.word_map = word_map
        self.vocab_size = len(word_map)
        if model is not None:
            self.model = model
        else:
            self.model = GridTDModel(args.embed_dim, args.hidden_dim, len(word_map), args.encoder)
            checkpoint = torch.load(args.weight)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.cuda()
        self.model.eval()

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.img_transform = transforms.Compose([transforms.Resize(size=(args.height, args.width)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=self.mean, std=self.std)])

        self.adalstm_weight_i = self.model.AdaLSTM.lstm_cell.weight_ih #(4*hidden_size, hiddendim+embed_dim)
        self.adalstm_weight_h = self.model.AdaLSTM.lstm_cell.weight_hh #(4*hidden_size, hiddendim)
        self.adalstm_bias_i = self.model.AdaLSTM.lstm_cell.bias_ih #(4*hidden_size,)
        self.adalstm_bias_h = self.model.AdaLSTM.lstm_cell.bias_hh #(4*hidden_size,)

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
        attribution = weight * forward_input #(out_feature, in_feature)
        # print(attribution.size())
        if type(forward_output) == bool:
            forward_output = torch.matmul(forward_input, weight.transpose(0,1))
        forward_output_eps = self.EPS * forward_output.sign() + forward_output  # Z.sign() returns -1 or 0 or 1

        forward_output_eps.masked_fill_(forward_output_eps == 0, self.EPS)  #(out_feature,)
        # print(forward_output_eps.size())
        attribution_norm = attribution.transpose(0,1) / forward_output_eps #(in_feature, out_feature)
        # print(attribution_norm.size())
        relevance_input = torch.sum(attribution_norm * r_out, dim=1) #(in_feature,)
        torch.cuda.empty_cache()
        return relevance_input

    def preprocess_img(self, img_filepath):
        image_data = Image.open(img_filepath).convert('RGB')
        img = self.img_transform(image_data)
        img = img.unsqueeze(0).cuda()
        return img

    def adalstm_forward(self, xt, ht_m1, ct_m1):
        z = torch.matmul(self.adalstm_weight_i, xt.squeeze())  #(4*hidden_size, 1)
        z = z + torch.matmul(self.adalstm_weight_h, ht_m1) #(4*hidden_size,1)
        z = z + self.adalstm_bias_h + self.adalstm_bias_i
        z0, z1, z2, z3 = z.chunk(4)
        i = torch.sigmoid(z0)
        f = torch.sigmoid(z1)
        c = f * ct_m1 + i * torch.tanh(z2)
        o = torch.sigmoid(z3)
        ht = o * torch.tanh(c)
        ct = c
        return ht, ct, z2, i, f  #(512, )

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

    def forward_greedy(self, img_filepath ):
        self.img = self.preprocess_img(img_filepath) #(bs, C, H, W)
        self.beam_caption, self.beam_caption_encode = self.model.beam_search(self.img, self.word_map, beam_size=1, max_cap_length=20)
        self.beam_caption_encode = [self.word_map['<start>']] + self.beam_caption_encode
        print(f'the predicted caption of {img_filepath} is "{self.beam_caption[0]}"')
        print(self.beam_caption_encode)
        # perform the forward pass and save the intermediate variables
        self.image_features, self.avg_feature = self.model.img_encoder(self.img)  # (bs, fea_dim, H, W), (bs, fea_dim)
        self.num_pixels = self.image_features.size(-1) * self.image_features.size(-2)
        self.image_feature_proj = self.model.relu(self.model.img_projector(self.image_features))  # (bs, hiddendim, H, W)
        self.global_img_feature = self.model.relu(self.model.global_img_feature_proj(self.avg_feature))  # (bs, embedding_dim)
        self.image_feature_proj = self.image_feature_proj.contiguous()
        self.image_feature_proj = self.image_feature_proj.view(1, self.model.hidden_dim, -1)  # (bs, hidden_dim, num_pixel)
        self.caption_length = len(self.beam_caption_encode) - 1
        self.predictions = torch.zeros(self.caption_length, self.vocab_size).cuda()
        self.x1t = torch.zeros(self.caption_length, 2 * self.model.embed_dim + self.model.hidden_dim).cuda()
        self.x2t = torch.zeros(self.caption_length, 2 * self.model.hidden_dim).cuda()
        self.betas = torch.zeros(self.caption_length).cuda()
        self.alphas = torch.zeros(self.caption_length, self.num_pixels).cuda()
        self.h1t = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.c1t = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.h2t = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.c2t = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.g1t = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.i1t_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.f1t_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.g2t = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.i2t_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.f2t_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.st = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context_hat = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        caption = [self.word_map['<start>']]
        for t in range(50):
            it = torch.LongTensor([caption[t]]).cuda()
            word_embedding = self.model.embedding(it)  # (1, embed_dim)
            if self.global_img_feature.dim() == 1:
                self.global_img_feature = self.global_img_feature.unsqueeze(0)
            h1t_m1 = self.h1t[t]
            c1t_m1 = self.c1t[t]
            h2t_m1 = self.h2t[t]
            c2t_m1 = self.c2t[t]
            x1_t = torch.cat((h2t_m1.unsqueeze(0), self.global_img_feature, word_embedding), dim=-1)   # (1, 2*embed_dim)
            h1_t, c1_t, g1_t, i1_t_act, f1_t_act = self.adalstm_forward(x1_t, h1t_m1, c1t_m1)
            sen_gate = torch.sigmoid(self.model.AdaLSTM.x_gate(x1_t) + self.model.AdaLSTM.h_gate(h1t_m1))
            s_t = sen_gate * torch.tanh(c1_t)
            context_t_hat, context_t, alpha_t, beta_t = self.model.AdaAttention(self.image_feature_proj, h1_t.unsqueeze(0), s_t) #(1, hidden_dim), (1, num_pixel), (1,1)
            x2_t = torch.cat((context_t_hat, h1_t.unsqueeze(0)), dim=-1)
            h2_t, c2_t, g2_t, i2_t_act, f2_t_act = self.language_lstm_forward(x2_t, h2t_m1, c2t_m1)
            predict_score_t = self.model.fc(context_t_hat + h2_t) #(1, vocab_size)
            label = torch.argmax(predict_score_t)
            if label == self.word_map['<end>']:
                print(caption)
                return
            caption.append(label)
            # here we save the intermediate states for further relevance backpropagation
            self.x1t[t] = x1_t.squeeze()
            print(x1_t.size())
            self.x2t[t] = x2_t.squeeze()
            print(x2_t.size())
            self.predictions[t,:] = predict_score_t.squeeze()
            print(predict_score_t.size())
            self.alphas[t] = alpha_t.squeeze()
            print(alpha_t.size())
            self.betas[t] = beta_t.squeeze()
            print(beta_t.size())
            self.h1t[t+1] = h1_t
            print(h1_t.size())
            self.c1t[t+1] = c1_t
            print(c1_t.size())
            self.g1t[t] = g1_t
            print(g1_t.size())
            self.i1t_act[t] = i1_t_act
            print(i1_t_act.size())
            self.f1t_act[t] = f1_t_act
            print(f1_t_act.size())
            self.h2t[t+1] = h2_t
            print(h2_t.size())
            self.c2t[t+1] = c2_t
            print(c2_t.size())
            self.g2t[t] = g2_t
            print(g2_t.size())
            self.i2t_act[t] = i2_t_act
            print(i2_t_act.size())
            self.f2t_act[t] = f2_t_act
            print(f2_t_act.size())
            self.st[t] = s_t.squeeze()
            print(s_t.size())
            self.context[t] = context_t.squeeze()
            print(context_t.size())
            self.context_hat[t] = context_t_hat.squeeze()
            print(context_t_hat.size())

    def teacherforce_forward(self, img, beam_caption_encode):
        # print(beam_caption_encode)
        # perform the forward pass and save the intermediate variables
        image_features, avg_feature = self.model.img_encoder(img)  # (bs, fea_dim, H, W), (bs, fea_dim)
        image_feature_proj = self.model.relu(self.model.img_projector(image_features))  # (bs, hiddendim, H, W)
        global_img_feature = self.model.relu(self.model.global_img_feature_proj(avg_feature))  # (bs, embedding_dim)
        image_feature_proj = image_feature_proj.contiguous()
        image_feature_proj = image_feature_proj.view(1, self.model.hidden_dim,-1)  # (bs, hidden_dim, num_pixel)
        caption_length = len(beam_caption_encode)
        predictions = torch.zeros(caption_length, self.vocab_size).cuda()
        h1t = torch.zeros(caption_length + 1, self.model.hidden_dim).cuda()
        c1t = torch.zeros(caption_length + 1, self.model.hidden_dim).cuda()
        h2t = torch.zeros(caption_length + 1, self.model.hidden_dim).cuda()
        c2t = torch.zeros(caption_length + 1, self.model.hidden_dim).cuda()
        for t in range(caption_length):
            it = torch.LongTensor([beam_caption_encode[t]]).cuda()
            word_embedding = self.model.embedding(it)  # (1, embed_dim)
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            h1t_m1 = h1t[t]
            c1t_m1 = c1t[t]
            h2t_m1 = h2t[t]
            c2t_m1 = c2t[t]
            x1_t = torch.cat((h2t_m1.unsqueeze(0), global_img_feature, word_embedding), dim=-1)  # (1, 2*embed_dim)
            h1_t, c1_t, g1_t, i1_t_act, f1_t_act = self.adalstm_forward(x1_t, h1t_m1, c1t_m1)
            sen_gate = torch.sigmoid(self.model.AdaLSTM.x_gate(x1_t) + self.model.AdaLSTM.h_gate(h1t_m1))
            s_t = sen_gate * torch.tanh(c1_t)
            context_t_hat, context_t, alpha_t, beta_t = self.model.AdaAttention(image_feature_proj,
                                                                                h1_t.unsqueeze(0),
                                                                                s_t)  # (1, hidden_dim), (1, num_pixel), (1,1)
            x2_t = torch.cat((context_t_hat, h1_t.unsqueeze(0)), dim=-1)
            h2_t, c2_t, g2_t, i2_t_act, f2_t_act = self.language_lstm_forward(x2_t, h2t_m1, c2t_m1)
            predict_score_t = self.model.fc(context_t_hat + h2_t)  # (1, vocab_size)
            # here we save the intermediate states for further relevance backpropagation
            predictions[t, :] = predict_score_t[0]
            h1t[t + 1] = h1_t
            c1t[t + 1] = c1_t
            h2t[t + 1] = h2_t
            c2t[t + 1] = c2_t
        return predictions

    def get_hidden_parameters(self, img_filepath ):
        self.img = self.preprocess_img(img_filepath)  # (bs, C, H, W)
        self.beam_caption, self.beam_caption_encode = self.model.beam_search(self.img, self.word_map, beam_size=2,
                                                                             max_cap_length=50)
        self.beam_caption_encode = [self.word_map['<start>']] + self.beam_caption_encode
        print(f'the predicted caption of {img_filepath} is "{self.beam_caption[0]}"')
        print(self.beam_caption_encode)
        # perform the forward pass and save the intermediate variables
        self.image_features, self.avg_feature = self.model.img_encoder(self.img)  # (bs, fea_dim, H, W), (bs, fea_dim)
        self.num_pixels = self.image_features.size(-1) * self.image_features.size(-2)
        # print(self.image_features.size(), self.num_pixels)
        self.image_feature_proj_before_act = self.model.img_projector(self.image_features)
        self.image_feature_proj = self.model.relu(self.image_feature_proj_before_act)  # (bs, hiddendim, H, W)
        self.global_img_feature_before_act = self.model.global_img_feature_proj(self.avg_feature)
        self.global_img_feature = self.model.relu(self.global_img_feature_before_act)  # (bs, embedding_dim)
        self.image_feature_proj = self.image_feature_proj.contiguous()
        self.image_feature_proj = self.image_feature_proj.view(1, self.model.hidden_dim, -1)  # (bs, hidden_dim, num_pixel)
        self.image_feature_proj_before_act = self.image_feature_proj_before_act.contiguous().view(1, self.model.hidden_dim, -1)

        self.caption_length = len(self.beam_caption_encode) - 1
        self.predictions = torch.zeros(self.caption_length, self.vocab_size).cuda()
        self.x1t = torch.zeros(self.caption_length, 2 * self.model.embed_dim + self.model.hidden_dim).cuda()
        self.x2t = torch.zeros(self.caption_length, 2 * self.model.hidden_dim).cuda()
        self.betas = torch.zeros(self.caption_length).cuda()
        self.alphas = torch.zeros(self.caption_length, self.num_pixels).cuda()
        self.h1t = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.c1t = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.h2t = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.c2t = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.g1t = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.i1t_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.f1t_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.g2t = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.i2t_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.f2t_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.st = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context_hat = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        for t in range(self.caption_length):
            it = torch.LongTensor([self.beam_caption_encode[t]]).cuda()
            word_embedding = self.model.embedding(it)  # (1, embed_dim)
            if self.global_img_feature.dim() == 1:
                self.global_img_feature = self.global_img_feature.unsqueeze(0)
            h1t_m1 = self.h1t[t]
            c1t_m1 = self.c1t[t]
            h2t_m1 = self.h2t[t]
            c2t_m1 = self.c2t[t]
            x1_t = torch.cat((h2t_m1.unsqueeze(0), self.global_img_feature, word_embedding), dim=-1)  # (1, 2*embed_dim)
            h1_t, c1_t, g1_t, i1_t_act, f1_t_act = self.adalstm_forward(x1_t, h1t_m1, c1t_m1)
            sen_gate = torch.sigmoid(self.model.AdaLSTM.x_gate(x1_t) + self.model.AdaLSTM.h_gate(h1t_m1))
            s_t = sen_gate * torch.tanh(c1_t)
            context_t_hat, context_t, alpha_t, beta_t = self.model.AdaAttention(self.image_feature_proj,
                                                                                h1_t.unsqueeze(0),
                                                                                s_t)  # (1, hidden_dim), (1, num_pixel), (1,1)
            x2_t = torch.cat((context_t_hat, h1_t.unsqueeze(0)), dim=-1)
            h2_t, c2_t, g2_t, i2_t_act, f2_t_act = self.language_lstm_forward(x2_t, h2t_m1, c2t_m1)

            predict_score_t = self.model.fc(context_t_hat + h2_t)  # (1, vocab_size)
            # print(x1_t.size(), predict_score_t.size(), alpha_t.size(), beta_t.size(), h1_t.size(), c1_t.size(), g1_t.size(),
            #       i1_t_act.size(), f1_t_act.size(),h2_t.size(), c2_t.size(), g2_t.size(),
            #       i2_t_act.size(), f2_t_act.size(), s_t.size(), context_t.size(), context_t_hat.size())
            # here we save the intermediate states for further relevance backpropagation
            self.x1t[t] = x1_t.squeeze()
            self.x2t[t] = x2_t.squeeze()
            self.predictions[t, :] = predict_score_t.squeeze()
            self.alphas[t] = alpha_t.squeeze()
            self.betas[t] = beta_t.squeeze()
            self.h1t[t + 1] = h1_t
            self.c1t[t + 1] = c1_t
            self.g1t[t] = g1_t
            self.i1t_act[t] = i1_t_act
            self.f1t_act[t] = f1_t_act
            self.h2t[t + 1] = h2_t
            self.c2t[t + 1] = c2_t
            self.g2t[t] = g2_t
            self.i2t_act[t] = i2_t_act
            self.f2t_act[t] = f2_t_act
            self.st[t] = s_t.squeeze()
            self.context[t] = context_t.squeeze()
            self.context_hat[t] = context_t_hat.squeeze()

    def explain_caption_wordt(self, t):
        assert t < self.caption_length  #(t starts from 0)
        preceeding_cap_length = t+1
        target_word_encode = self.beam_caption_encode[t+1]
        words = self.beam_caption[0].split(' ')
        weight_ig = self.adalstm_weight_i.chunk(4,0)[2]  #(hidden_dim, embed_dim*2 + hidden_dim)
        weight_hg = self.adalstm_weight_h.chunk(4,0)[2]  #(hidden_dim, hidden_dim)
        weight_g = torch.cat((weight_ig, weight_hg), dim=1) #(hidden_dim, 2 * hidden_dim+2*embed_dim)
        language_weight_ig = self.language_weight_i.chunk(4,0)[2]  #(hidden_dim, 2*hidden_dim)
        language_weight_hg = self.language_weight_h.chunk(4,0)[2]  #(hidden_dim, hidden_dim)
        language_weight_g = torch.cat((language_weight_ig, language_weight_hg), dim=-1) #(hidden_dim, 3 * hidden_dim)
        xh1t = torch.cat((self.x1t[:preceeding_cap_length], self.h1t[:preceeding_cap_length]), dim=1) #(preceeding_length, 2*hidden_dim+2*embed_dim)
        xh2t = torch.cat((self.x2t[:preceeding_cap_length], self.h2t[:preceeding_cap_length]), dim=1) #(preceeding_length, 3*hidden_dim)
        predict_score_t = self.predictions[t] #(vocat_size,)
        # print(target_word_encode, words[t], t + 1, torch.argmax(predict_score_t), torch.max(predict_score_t), predict_score_t[target_word_encode])
        image_feature = self.image_features.view(1, self.model.encoder_raw_dim, self.num_pixels)
        image_feature = image_feature.transpose(1,2).squeeze(0) #(num_pixel, encode_raw_dim)
        image_feature_proj = self.image_feature_proj.transpose(1,2).squeeze(0) #(num_pixel, hidden_dim)
        image_feature_proj_before_act = self.image_feature_proj_before_act.transpose(1,2).squeeze(0)
        word_relevance = torch.zeros(1, self.vocab_size).cuda()
        word_relevance[0, target_word_encode] = predict_score_t[target_word_encode]
        self.r_h1t = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        self.r_c1t = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        self.r_h2t = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        self.r_c2t = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        self.r_xh1t = torch.zeros(preceeding_cap_length, 2 * self.model.hidden_dim + 2 * self.model.embed_dim).cuda()
        self.r_xh2t = torch.zeros(preceeding_cap_length, 3 * self.model.hidden_dim).cuda()
        self.r_global_img_feature = torch.zeros(self.model.embed_dim).cuda()
        self.r_word_embedding = torch.zeros(preceeding_cap_length, self.model.embed_dim).cuda()
        self.r_img_feature = torch.zeros(self.num_pixels, self.model.encoder_raw_dim).cuda()
        self.r_img_feature_proj = torch.zeros(self.num_pixels, self.model.hidden_dim).cuda()
        self.r_context = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        self.r_context_hat = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        r_h2t_context = self.lrp_linear_eps(r_out=word_relevance,
                                           forward_input=self.h2t[t+1]+self.context_hat[t],
                                           forward_output=predict_score_t,
                                           weight=self.output_weight)
        self.r_h2t[t+1] = self.lrp_linear_eps(r_out=r_h2t_context,
                                             forward_input=self.h2t[t+1],
                                             forward_output=self.h2t[t+1]+self.context_hat[t],
                                             weight=torch.eye(self.model.hidden_dim).cuda())

        self.r_context_hat[t] = self.lrp_linear_eps(r_out=r_h2t_context,
                                                    forward_input=self.context_hat[t],
                                                    forward_output=self.h2t[t+1]+self.context_hat[t],
                                                    weight=torch.eye(self.model.hidden_dim).cuda())
        for i in range(t+1)[::-1]:
            self.r_c2t[i+1] = self.r_c2t[i+1] + self.r_h2t[i+1]
            r_g2t = self.lrp_linear_eps(r_out=self.r_c2t[i + 1],
                                       forward_input=self.i2t_act[i] * torch.tanh(self.g2t[i]),
                                       forward_output=self.c2t[i+1],
                                       weight=torch.eye(self.model.hidden_dim).cuda())
            self.r_c2t[i] = self.lrp_linear_eps(r_out=self.r_c2t[i + 1],
                                                forward_input=self.f2t_act[i] * self.c2t[i],
                                                forward_output=self.c2t[i+1],
                                                weight=torch.eye(self.model.hidden_dim).cuda())
            self.r_xh2t[i] = self.lrp_linear_eps(r_out=r_g2t,
                                                forward_input=xh2t[i],
                                                forward_output=self.g2t[i],
                                                weight=language_weight_g)
            self.r_h2t[i] = self.r_xh2t[i][self.model.hidden_dim*2:]
            self.r_h1t[i+1] = self.r_xh2t[i][self.model.hidden_dim:2*self.model.hidden_dim]
            self.r_context_hat[i] += self.r_xh2t[i][:self.model.hidden_dim]
            r_st = self.lrp_linear_eps(r_out=self.r_context_hat[i],
                                       forward_input=self.betas[i] * self.st[i],
                                       forward_output=self.context_hat[i],
                                       weight=torch.eye(self.model.hidden_dim).cuda())
            self.r_context[i] = self.lrp_linear_eps(r_out=self.r_context_hat[i],
                                                    forward_input=self.context[i]*(1-self.betas[i]),
                                                    forward_output=self.context_hat[i],
                                                    weight=torch.eye(self.model.hidden_dim).cuda())
            # if i == t:
            #     for k in range(self.num_pixels):
            #         self.r_img_feature_proj[k] += self.lrp_linear_eps(r_out=self.r_context[i],
            #                                                           forward_input=image_feature_proj[k]*self.alphas[i][k],
            #                                                           forward_output=self.context[i],
            #                                                           weight=torch.eye(self.model.hidden_dim).cuda())
            for k in range(self.num_pixels):
                self.r_img_feature_proj[k] += self.lrp_linear_eps(r_out=self.r_context[i],
                                                                  forward_input=image_feature_proj[k] * self.alphas[i][k],
                                                                  forward_output=self.context[i],
                                                                  weight=torch.eye(self.model.hidden_dim).cuda())
            self.r_c1t[i+1] += r_st
            self.r_c1t[i+1] += self.r_h1t[i+1]
            r_g1t = self.lrp_linear_eps(r_out=self.r_c1t[i+1],
                                        forward_input=self.i1t_act[i] * torch.tanh(self.g1t[i]),
                                        forward_output=self.c1t[i+1],
                                        weight=torch.eye(self.model.hidden_dim).cuda())
            self.r_c1t[i] = self.lrp_linear_eps(r_out=self.r_c1t[i+1],
                                                forward_input=self.f1t_act[i] * self.c1t[i],
                                                forward_output=self.c1t[i+1],
                                                weight=torch.eye(self.model.hidden_dim).cuda())
            self.r_xh1t[i] = self.lrp_linear_eps(r_out=r_g1t,
                                                 forward_input=xh1t[i],
                                                 forward_output=self.g1t[i],
                                                 weight=weight_g)
            self.r_h1t[i] = self.r_xh1t[i][2*self.model.embed_dim+self.model.hidden_dim:]
            self.r_h2t[i] += self.r_xh1t[i][:self.model.hidden_dim]
            # if i == t:
            #     self.r_global_img_feature = self.r_global_img_feature + self.r_xh1t[i][self.model.hidden_dim:self.model.embed_dim+self.model.hidden_dim]
            self.r_global_img_feature = self.r_global_img_feature + self.r_xh1t[i][self.model.hidden_dim:self.model.embed_dim + self.model.hidden_dim]
            self.r_word_embedding[i] = self.r_xh1t[i][self.model.hidden_dim+self.model.embed_dim:self.model.embed_dim*2+self.model.hidden_dim]
        r_average_img_feature = self.lrp_linear_eps(r_out=self.r_global_img_feature,
                                                    forward_input=self.avg_feature,
                                                    forward_output=self.global_img_feature_before_act,
                                                    weight=self.model.global_img_feature_proj.weight)
        for i in range(self.num_pixels):
            self.r_img_feature[i] = self.lrp_linear_eps(r_out=r_average_img_feature,
                                                        forward_input=image_feature[i]/self.num_pixels,
                                                        forward_output=self.avg_feature,
                                                        weight=torch.eye(self.model.encoder_raw_dim).cuda())
            self.r_img_feature[i] = self.r_img_feature[i] + self.lrp_linear_eps(r_out=self.r_img_feature_proj[i],
                                                                                forward_input=image_feature[i],
                                                                                forward_output=image_feature_proj_before_act[i],
                                                                                weight=self.model.img_projector.weight.squeeze(-1).squeeze(-1))
        r_words = torch.sum(self.r_word_embedding, dim=-1)
        max_abs_r_words = torch.max(torch.abs(r_words))
        if max_abs_r_words > 0:
            r_words = r_words / max_abs_r_words
        r_img_feature = self.r_img_feature.unsqueeze(0).transpose(1,2).view(self.image_features.size())
        torch.cuda.empty_cache()
        return r_img_feature, r_words

    def explain_cnn(self, r_img_feature):
        relevance_img = self.model.img_encoder.encoder.compute_lrp(self.img, target=r_img_feature)
        return relevance_img

    def explain_caption(self, img_filepath, t_list=None):
        self.img_filepath = img_filepath
        self.get_hidden_parameters(img_filepath)
        relevance_imgs = []
        relevance_preceeding_words = []
        lrp_wrapper.add_lrp(self.model.img_encoder.encoder)
        for t in range(self.caption_length):
            with torch.no_grad():
                relevance_img_feature, r_words = self.explain_caption_wordt(t)
            relevance_img = self.explain_cnn(relevance_img_feature)
            relevance_imgs.append(relevance_img)
            relevance_preceeding_words.append(r_words)
        assert len(relevance_imgs) == self.caption_length
        self.visualize_explanations(relevance_imgs, t=t_list)
        self.save_linguistic_explanation(relevance_preceeding_words)
        return relevance_imgs, relevance_preceeding_words

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

    def visualize_explanations(self, relevance_imgs, t=None):
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
                hm_show.save(os.path.join(save_dir, str(i) + '_lrp_' + words[i] + '.jpg'))
            axes[i].set_title(words[i], fontsize=18)
            axes[i].imshow(hm_show)
        plt.savefig(os.path.join(save_dir, 'lrp_hm.jpg'))
        _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20, 20))
        axes = axes.flatten()
        for i in range(self.caption_length):
            # print(self.alphas[i].size())
            atten_hm = LRPutil.visuallize_attention(img_original, self.alphas[i],
                                                    (int(np.sqrt(self.num_pixels)),int(np.sqrt(self.num_pixels))), upscale=16)
            if isinstance(t, list) and  i in t:
                atten_hm.save(os.path.join(save_dir, str(i) + '_attention_' + words[i] + '.jpg'))
            axes[i].set_title(words[i], fontsize=18)
            axes[i].imshow(atten_hm)
        plt.savefig(os.path.join(save_dir, 'attention_hm.jpg'))


class ExplainGridTDGradient(object):
    EX_TYPE = 'gradient'
    def __init__(self, args, word_map):
        super(ExplainGridTDGradient, self).__init__()
        self.args = args
        self.word_map = word_map
        self.vocab_size = len(word_map)
        self.model = GridTDModel(args.embed_dim, args.hidden_dim, len(word_map), args.encoder)
        checkpoint = torch.load(args.weight)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.cuda()
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.img_transform = transforms.Compose([transforms.Resize(size=(args.height, args.width)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=self.mean, std=self.std)])

        self.adalstm_weight_i = self.model.AdaLSTM.lstm_cell.weight_ih #(4*hidden_size, hiddendim+embed_dim)
        self.adalstm_weight_h = self.model.AdaLSTM.lstm_cell.weight_hh #(4*hidden_size, hiddendim)
        self.adalstm_bias_i = self.model.AdaLSTM.lstm_cell.bias_ih #(4*hidden_size,)
        self.adalstm_bias_h = self.model.AdaLSTM.lstm_cell.bias_hh #(4*hidden_size,)

        self.language_weight_i = self.model.LanguageLSTM.weight_ih #(4*hidden_size, hiddendim+embed_dim)
        self.language_weight_h = self.model.LanguageLSTM.weight_hh #(4*hidden_size, hiddendim)
        self.language_bias_i = self.model.LanguageLSTM.bias_ih #(4*hidden_size,)
        self.language_bias_h = self.model.LanguageLSTM.bias_hh #(4*hidden_size,)

        self.output_weight = self.model.fc.weight  #(vocab_size, hidden_dim)

        self.visualizatioin_save_path = os.path.join(args.save_path, args.dataset + 'explanation')
        if not os.path.isdir(self.visualizatioin_save_path):
            os.makedirs(self.visualizatioin_save_path)

    def adalstm_forward(self, xt, ht_m1, ct_m1):
        z = torch.matmul(self.adalstm_weight_i, xt.squeeze(0))  #(4*hidden_size, )
        z = z + torch.matmul(self.adalstm_weight_h, ht_m1) #(4*hidden_size,)
        z = z + self.adalstm_bias_h + self.adalstm_bias_i
        z0, z1, z2, z3 = z.chunk(4)
        i = torch.sigmoid(z0)
        f = torch.sigmoid(z1)
        g = torch.tanh(z2)
        c = f * ct_m1 + i * torch.tanh(z2)
        o = torch.sigmoid(z3)
        ht = o * torch.tanh(c)
        ct = c
        return ht, ct, z0, z1, z2, z3, i, f, g, o

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

    def preprocess_img(self, img_filepath):
        image_data = Image.open(img_filepath).convert('RGB')
        img = self.img_transform(image_data)
        img = img.unsqueeze(0).cuda()
        return img

    def teacherforce_forward(self, img, beam_caption_encode):
        # print(beam_caption_encode)
        # perform the forward pass and save the intermediate variables
        image_features, avg_feature = self.model.img_encoder(img)  # (bs, fea_dim, H, W), (bs, fea_dim)
        image_feature_proj = self.model.relu(self.model.img_projector(image_features))  # (bs, hiddendim, H, W)
        global_img_feature = self.model.relu(self.model.global_img_feature_proj(avg_feature))  # (bs, embedding_dim)
        image_feature_proj = image_feature_proj.contiguous()
        image_feature_proj = image_feature_proj.view(1, self.model.hidden_dim,-1)  # (bs, hidden_dim, num_pixel)
        caption_length = len(beam_caption_encode)
        predictions = torch.zeros(caption_length, self.vocab_size).cuda()
        h1t = torch.zeros(caption_length + 1, self.model.hidden_dim).cuda()
        c1t = torch.zeros(caption_length + 1, self.model.hidden_dim).cuda()
        h2t = torch.zeros(caption_length + 1, self.model.hidden_dim).cuda()
        c2t = torch.zeros(caption_length + 1, self.model.hidden_dim).cuda()
        for t in range(caption_length):
            it = torch.LongTensor([beam_caption_encode[t]]).cuda()
            word_embedding = self.model.embedding(it)  # (1, embed_dim)
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            h1t_m1 = h1t[t]
            c1t_m1 = c1t[t]
            h2t_m1 = h2t[t]
            c2t_m1 = c2t[t]
            x1_t = torch.cat((h2t_m1.unsqueeze(0), global_img_feature, word_embedding), dim=-1)  # (1, 2*embed_dim)
            h1_t, c1_t, i1_t, f1_t, g1_t, o1_t, i1_t_act, f1_t_act, g1_t_act, o1_t_act = self.adalstm_forward(x1_t, h1t_m1, c1t_m1)
            sen_gate = torch.sigmoid(self.model.AdaLSTM.x_gate(x1_t) + self.model.AdaLSTM.h_gate(h1t_m1))
            s_t = sen_gate * torch.tanh(c1_t)
            context_t_hat, context_t, alpha_t, beta_t = self.model.AdaAttention(image_feature_proj,
                                                                                h1_t.unsqueeze(0),
                                                                                s_t)  # (1, hidden_dim), (1, num_pixel), (1,1)
            x2_t = torch.cat((context_t_hat, h1_t.unsqueeze(0)), dim=-1)
            h2_t, c2_t, i2_t, f2_t, g2_t, o2_t, i2_t_act, f2_t_act, g2_t_act, o2_t_act = self.language_lstm_forward(x2_t, h2t_m1, c2t_m1)
            predict_score_t = self.model.fc(context_t_hat + h2_t)  # (1, vocab_size)
            # here we save the intermediate states for further relevance backpropagation
            predictions[t, :] = predict_score_t[0]
            h1t[t + 1] = h1_t
            c1t[t + 1] = c1_t
            h2t[t + 1] = h2_t
            c2t[t + 1] = c2_t
        return predictions

    def get_hidden_parameters(self, img_filepath):
        self.img = self.preprocess_img(img_filepath)  # (bs, C, H, W)
        self.beam_caption, self.beam_caption_encode = self.model.beam_search(self.img, self.word_map, beam_size=3,
                                                                             max_cap_length=50)
        self.beam_caption_encode = [self.word_map['<start>']] + self.beam_caption_encode
        print(f'the predicted caption of {img_filepath} is "{self.beam_caption[0]}"')
        print(self.beam_caption_encode)
        # perform the forward pass and save the intermediate variables
        self.image_features, self.avg_feature = self.model.img_encoder(self.img)  # (bs, fea_dim, H, W), (bs, fea_dim)
        self.num_pixels = self.image_features.size(-1) * self.image_features.size(-2)
        self.image_feature_proj = self.model.relu(
            self.model.img_projector(self.image_features))  # (bs, hiddendim, H, W)
        self.global_img_feature = self.model.relu(
            self.model.global_img_feature_proj(self.avg_feature))  # (bs, embedding_dim)
        self.image_feature_proj = self.image_feature_proj.contiguous()
        self.image_feature_proj = self.image_feature_proj.view(1, self.model.hidden_dim,
                                                               -1)  # (bs, hidden_dim, num_pixel)
        self.caption_length = len(self.beam_caption_encode) - 1
        self.predictions = torch.zeros(self.caption_length, self.vocab_size).cuda()
        self.x1t = torch.zeros(self.caption_length, 2 * self.model.embed_dim + self.model.hidden_dim).cuda()
        self.x2t = torch.zeros(self.caption_length, 2 * self.model.hidden_dim).cuda()
        self.betas = torch.zeros(self.caption_length).cuda()
        self.alphas = torch.zeros(self.caption_length, self.num_pixels).cuda()
        self.h1t = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.c1t = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.h2t = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.c2t = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.i1t = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.f1t = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.g1t = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.o1t = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.i1t_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.f1t_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.g1t_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.o1t_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.i2t = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.f2t = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.g2t = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.o2t = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.i2t_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.f2t_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.g2t_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.o2t_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.st = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.sen_gate = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context_hat = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        for t in range(self.caption_length):
            it = torch.LongTensor([self.beam_caption_encode[t]]).cuda()
            word_embedding = self.model.embedding(it)  # (1, embed_dim)
            if self.global_img_feature.dim() == 1:
                self.global_img_feature = self.global_img_feature.unsqueeze(0)
            h1t_m1 = self.h1t[t]
            c1t_m1 = self.c1t[t]
            h2t_m1 = self.h2t[t]
            c2t_m1 = self.c2t[t]
            x1_t = torch.cat((h2t_m1.unsqueeze(0), self.global_img_feature, word_embedding), dim=-1)  # (1, 2*embed_dim)
            h1_t, c1_t, i1_t, f1_t, g1_t, o1_t, i1_t_act, f1_t_act, g1_t_act, o1_t_act = self.adalstm_forward(x1_t, h1t_m1, c1t_m1)
            sen_gate_t = torch.sigmoid(self.model.AdaLSTM.x_gate(x1_t) + self.model.AdaLSTM.h_gate(h1t_m1))
            s_t = sen_gate_t * torch.tanh(c1_t)
            context_t_hat, context_t, alpha_t, beta_t = self.model.AdaAttention(self.image_feature_proj,
                                                                                h1_t.unsqueeze(0),
                                                                                s_t)  # (1, hidden_dim), (1, num_pixel), (1,1)
            x2_t = torch.cat((context_t_hat, h1_t.unsqueeze(0)), dim=-1)
            h2_t, c2_t, i2_t, f2_t, g2_t, o2_t, i2_t_act, f2_t_act, g2_t_act, o2_t_act = self.language_lstm_forward(x2_t, h2t_m1, c2t_m1)

            predict_score_t = self.model.fc(context_t_hat + h2_t)  # (1, vocab_size)
            # print(x1_t.size(), predict_score_t.size(), alpha_t.size(), beta_t.size(), h1_t.size(), c1_t.size(), g1_t.size(),
            #       i1_t_act.size(), f1_t_act.size(),h2_t.size(), c2_t.size(), g2_t.size(),
            #       i2_t_act.size(), f2_t_act.size(), s_t.size(), context_t.size(), context_t_hat.size())
            # here we save the intermediate states for further relevance backpropagation
            self.sen_gate[t] = sen_gate_t.squeeze()
            self.x1t[t] = x1_t.squeeze()
            self.x2t[t] = x2_t.squeeze()
            self.predictions[t, :] = predict_score_t.squeeze()
            self.alphas[t] = alpha_t.squeeze()
            self.betas[t] = beta_t.squeeze()
            self.h1t[t + 1] = h1_t
            self.c1t[t + 1] = c1_t
            self.i1t[t] = i1_t
            self.f1t[t] = f1_t
            self.g1t[t] = g1_t
            self.o1t[t] = o1_t
            self.i1t_act[t] = i1_t_act
            self.f1t_act[t] = f1_t_act
            self.g1t_act[t] = g1_t_act
            self.o1t_act[t] = o1_t_act
            self.h2t[t + 1] = h2_t
            self.c2t[t + 1] = c2_t
            self.i2t[t] = i2_t
            self.f2t[t] = f2_t
            self.g2t[t] = g2_t
            self.o2t[t] = o2_t
            self.i2t_act[t] = i2_t_act
            self.f2t_act[t] = f2_t_act
            self.g2t_act[t] = g2_t_act
            self.o2t_act[t] = o2_t_act
            self.st[t] = s_t.squeeze()
            self.context[t] = context_t.squeeze()
            self.context_hat[t] = context_t_hat.squeeze()

    def explain_caption_wordt(self,t):
        assert t < self.caption_length  #(t starts from 0)
        preceeding_cap_length = t+1
        target_word_encode = self.beam_caption_encode[t+1]
        d_word_pred = torch.zeros(1, self.vocab_size).cuda()
        d_word_pred[0, target_word_encode] = 1.  #(1, vocab_size)
        d_h1t = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        d_c1t = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        d_i1t = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_f1t = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_g1t = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_o1t = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_i1t_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_f1t_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_g1t_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_o1t_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_h2t = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        d_c2t = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        d_i2t = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_f2t = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_g2t = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_o2t = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_i2t_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_f2t_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_g2t_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_o2t_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_x1t = torch.zeros(preceeding_cap_length, self.model.hidden_dim + 2*self.model.embed_dim).cuda()
        d_x2t = torch.zeros(preceeding_cap_length, 2*self.model.hidden_dim ).cuda()
        d_context_hat = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_st = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_global_img_feature = torch.zeros(self.model.embed_dim).cuda()
        d_word_embedding = torch.zeros(preceeding_cap_length, self.model.embed_dim).cuda()
        d_img_feature = torch.zeros(self.num_pixels, self.model.encoder_raw_dim).cuda()
        d_img_feature_proj = torch.zeros(self.num_pixels, self.model.hidden_dim).cuda()
        #backward starts
        d_h2t_context_hat = torch.matmul(d_word_pred, self.output_weight).squeeze()
        d_context_hat[t] = d_h2t_context_hat * 1
        d_h2t[t+1] = d_h2t_context_hat * 1
        for i in range(preceeding_cap_length)[::-1]:
            d_o2t_act[i] = d_h2t[i+1] * torch.tanh(self.c2t[i+1])
            d_c2t[i+1] = d_c2t[i+1] + d_h2t[i+1] * self.o2t_act[i] * (1-(torch.tanh(self.c2t[i+1]))**2)
            d_f2t_act[i] = d_c2t[i+1] * self.c2t[i]
            d_c2t[i] = d_c2t[i+1] * self.f2t_act[i]
            d_i2t_act[i] = d_c2t[i+1] * self.g2t_act[i]
            d_g2t_act[i] = d_c2t[i+1] * self.i2t_act[i]
            d_i2t[i] = d_i2t_act[i] * self.i2t_act[i] * (1 - self.i2t_act[i])
            d_f2t[i] = d_f2t_act[i] * self.f2t_act[i] * (1 - self.f2t_act[i])
            d_o2t[i] = d_o2t_act[i] * self.o2t_act[i] * (1 - self.o2t_act[i])
            d_g2t[i] = d_g2t_act[i] * (1 - (self.g2t_act[i]) ** 2)
            d_gates2 = torch.cat((d_i2t[i: i+1], d_f2t[i: i+1], d_g2t[i: i+1], d_o2t[i: i+1]),dim=1) #(1, 4*hidden_dim)
            d_h2t[i] = torch.matmul(d_gates2, self.language_weight_h).squeeze() #(hidden_dim)
            d_x2t[i] = torch.matmul(d_gates2, self.language_weight_i).squeeze() #(2*embed_dim)
            d_context_hat[i] += d_x2t[i][: self.model.hidden_dim]
            d_context = d_context_hat[i] * (1 - self.betas[i])
            for k in range(self.num_pixels):
                d_img_feature_proj[k] += d_context * self.alphas[i][k]
            d_st[i] = d_context_hat[i] * self.betas[i]
            d_c1t[i + 1] += d_st[i] * self.sen_gate[i] * (1 - (torch.tanh(self.c1t[i + 1])) ** 2)
            d_h1t[i+1] = d_x2t[i][self.model.hidden_dim:]
            d_o1t_act[i] = d_h1t[i+1]*torch.tanh(self.c1t[i+1])
            d_c1t[i+1] = d_c1t[i+1] + d_h1t[i+1]*self.o1t_act[i] * (1-(torch.tanh(self.c1t[i+1]))**2)
            d_f1t_act[i] = d_c1t[i + 1] * self.c1t[i]
            d_c1t[i] = d_c1t[i + 1] * self.f1t_act[i]
            d_i1t_act[i] = d_c1t[i + 1] * self.g1t_act[i]
            d_g1t_act[i] = d_c1t[i + 1] * self.i1t_act[i]
            d_i1t[i] = d_i1t_act[i] * self.i1t_act[i] * (1 - self.i1t_act[i])
            d_f1t[i] = d_f1t_act[i] * self.f1t_act[i] * (1 - self.f1t_act[i])
            d_o1t[i] = d_o1t_act[i] * self.o1t_act[i] * (1 - self.o1t_act[i])
            d_g1t[i] = d_g1t_act[i] * (1 - (self.g1t_act[i]) ** 2)
            d_gates1 = torch.cat((d_i1t[i: i + 1], d_f1t[i: i + 1], d_g1t[i: i + 1], d_o1t[i: i + 1]),dim=1)
            d_h1t[i] = torch.matmul(d_gates1, self.adalstm_weight_h).squeeze()
            d_x1t[i] = torch.matmul(d_gates1, self.adalstm_weight_i).squeeze()
            d_global_img_feature = d_global_img_feature + d_x1t[i][self.model.hidden_dim:self.model.hidden_dim+self.model.embed_dim] #(embed_dim, )
            d_word_embedding[i] = d_x1t[i][self.model.hidden_dim + self.model.embed_dim:]
            d_h2t[i] += d_x1t[i][:self.model.hidden_dim]
        d_average_img_feature = torch.matmul(d_global_img_feature, self.model.global_img_feature_proj.weight).squeeze()
        for i in range(self.num_pixels):
            d_img_feature[i] = 1.0 * d_average_img_feature / self.num_pixels
            d_img_feature[i] = d_img_feature[i] + torch.matmul(d_img_feature_proj[i], self.model.img_projector.weight.squeeze(-1).squeeze(-1)).squeeze()
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
        del cnn_encoder
        return result

    def explain_caption(self, img_filepath, t_list=None):
        self.img_filepath = img_filepath
        self.get_hidden_parameters(img_filepath)
        self.image_feature_proj = self.image_feature_proj.transpose(1, 2)
        relevance_imgs = []
        relevance_preceeding_words = []
        for t in range(self.caption_length):
            relevance_img_feature, r_words = self.explain_caption_wordt(t)
            relevance_img = self.explain_cnn(relevance_img_feature)
            relevance_imgs.append(relevance_img)
            relevance_preceeding_words.append(r_words)
        assert len(relevance_imgs) == self.caption_length
        # self.visualize_explanations(relevance_imgs, t=t_list)
        # self.save_linguistic_explanation(relevance_preceeding_words)
        return relevance_imgs, relevance_preceeding_words

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

    def visualize_explanations(self, relevance_imgs, t=None):
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
                hm.save(os.path.join(save_dir, str(i) + '_gradient_' + words[i] + '.jpg'))
            axes[i].set_title(words[i], fontsize=18)
            axes[i].imshow(hm)
        plt.savefig(os.path.join(save_dir,'gradient_hm.jpg'))


class ExplainiGridTDGuidedGradient(ExplainGridTDGradient):
    EX_TYPE = 'GuidedBackpropagate'

    def explain_caption_wordt(self,t):
        assert t < self.caption_length  #(t starts from 0)
        preceeding_cap_length = t+1
        target_word_encode = self.beam_caption_encode[t+1]
        d_word_pred = torch.zeros(1, self.vocab_size).cuda()
        d_word_pred[0, target_word_encode] = 1.  #(1, vocab_size)
        d_h1t = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        d_c1t = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        d_i1t = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_f1t = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_g1t = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_o1t = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_i1t_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_f1t_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_g1t_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_o1t_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_h2t = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        d_c2t = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        d_i2t = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_f2t = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_g2t = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_o2t = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_i2t_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_f2t_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_g2t_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_o2t_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_x1t = torch.zeros(preceeding_cap_length, self.model.hidden_dim + 2*self.model.embed_dim).cuda()
        d_x2t = torch.zeros(preceeding_cap_length, 2*self.model.hidden_dim ).cuda()
        d_context_hat = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_st = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_global_img_feature = torch.zeros(self.model.embed_dim).cuda()
        d_word_embedding = torch.zeros(preceeding_cap_length, self.model.embed_dim).cuda()
        d_img_feature = torch.zeros(self.num_pixels, self.model.encoder_raw_dim).cuda()
        d_img_feature_proj = torch.zeros(self.num_pixels, self.model.hidden_dim).cuda()
        #backward starts
        d_h2t_context_hat = torch.matmul(d_word_pred, self.output_weight).squeeze()
        d_context_hat[t] = d_h2t_context_hat * 1
        d_h2t[t+1] = d_h2t_context_hat * 1
        for i in range(preceeding_cap_length)[::-1]:
            d_o2t_act[i] = d_h2t[i+1] * torch.tanh(self.c2t[i+1])
            d_c2t[i+1] = d_c2t[i+1] + d_h2t[i+1] * self.o2t_act[i] * (1-(torch.tanh(self.c2t[i+1]))**2)
            d_f2t_act[i] = d_c2t[i+1] * self.c2t[i]
            d_c2t[i] = d_c2t[i+1] * self.f2t_act[i]
            d_i2t_act[i] = d_c2t[i+1] * self.g2t_act[i]
            d_g2t_act[i] = d_c2t[i+1] * self.i2t_act[i]
            d_i2t[i] = d_i2t_act[i] * self.i2t_act[i] * (1 - self.i2t_act[i])
            d_f2t[i] = d_f2t_act[i] * self.f2t_act[i] * (1 - self.f2t_act[i])
            d_o2t[i] = d_o2t_act[i] * self.o2t_act[i] * (1 - self.o2t_act[i])
            d_g2t[i] = d_g2t_act[i] * (1 - (self.g2t_act[i]) ** 2)
            d_gates2 = torch.cat((d_i2t[i: i+1], d_f2t[i: i+1], d_g2t[i: i+1], d_o2t[i: i+1]),dim=1) #(1, 4*hidden_dim)
            d_h2t[i] = torch.matmul(d_gates2, self.language_weight_h).squeeze() #(hidden_dim)
            d_x2t[i] = torch.matmul(d_gates2, self.language_weight_i).squeeze() #(2*embed_dim)
            d_context_hat[i] += d_x2t[i][: self.model.hidden_dim]
            d_context = d_context_hat[i] * (1 - self.betas[i])
            for k in range(self.num_pixels):
                d_img_feature_proj[k] += d_context * self.alphas[i][k]
            d_st[i] = d_context_hat[i] * self.betas[i]
            d_c1t[i + 1] += d_st[i] * self.sen_gate[i] * (1 - (torch.tanh(self.c1t[i + 1])) ** 2)
            d_h1t[i+1] = d_x2t[i][self.model.hidden_dim:]
            d_o1t_act[i] = d_h1t[i+1]*torch.tanh(self.c1t[i+1])
            d_c1t[i+1] = d_c1t[i+1] + d_h1t[i+1]*self.o1t_act[i] * (1-(torch.tanh(self.c1t[i+1]))**2)
            d_f1t_act[i] = d_c1t[i + 1] * self.c1t[i]
            d_c1t[i] = d_c1t[i + 1] * self.f1t_act[i]
            d_i1t_act[i] = d_c1t[i + 1] * self.g1t_act[i]
            d_g1t_act[i] = d_c1t[i + 1] * self.i1t_act[i]
            d_i1t[i] = d_i1t_act[i] * self.i1t_act[i] * (1 - self.i1t_act[i])
            d_f1t[i] = d_f1t_act[i] * self.f1t_act[i] * (1 - self.f1t_act[i])
            d_o1t[i] = d_o1t_act[i] * self.o1t_act[i] * (1 - self.o1t_act[i])
            d_g1t[i] = d_g1t_act[i] * (1 - (self.g1t_act[i]) ** 2)
            d_gates1 = torch.cat((d_i1t[i: i + 1], d_f1t[i: i + 1], d_g1t[i: i + 1], d_o1t[i: i + 1]),dim=1)
            d_h1t[i] = torch.matmul(d_gates1, self.adalstm_weight_h).squeeze()
            d_x1t[i] = torch.matmul(d_gates1, self.adalstm_weight_i).squeeze()
            d_global_img_feature = d_global_img_feature + d_x1t[i][self.model.hidden_dim:self.model.hidden_dim+self.model.embed_dim] #(embed_dim, )
            d_word_embedding[i] = d_x1t[i][self.model.hidden_dim + self.model.embed_dim:]
            d_h2t[i] += d_x1t[i][:self.model.hidden_dim]
        d_global_img_feature[self.global_img_feature[0]<0] = 0
        d_average_img_feature = torch.matmul(d_global_img_feature, self.model.global_img_feature_proj.weight).squeeze()
        d_img_feature_proj[self.image_feature_proj[0] < 0] = 0
        for i in range(self.num_pixels):
            d_img_feature[i] = 1.0 * d_average_img_feature / self.num_pixels
            d_img_feature[i] = d_img_feature[i] + torch.matmul(d_img_feature_proj[i], self.model.img_projector.weight.squeeze(-1).squeeze(-1)).squeeze()
        r_words = torch.sum(d_word_embedding, dim=-1)
        max_abs_r_words = torch.max(torch.abs(r_words))
        if max_abs_r_words > 0:
            r_words = r_words / max_abs_r_words
        d_img_feature = d_img_feature.unsqueeze(0).transpose(1,2).view(self.image_features.size()) #(bs, C, H, W)
        d_img_feature[self.image_features<=0] = 0
        return d_img_feature, r_words

    def register_hooks(self, model):
        def forward_hook_fn(module, input, output):
            module.output_ = output
        def backward_hook_fn(module, grad_in, grad_out):
            grad = module.output_
            grad[grad>0] = 1
            grad[grad<=0] = 0
            positive_grad_out = torch.clamp(grad_out[0], min=0.0)
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
        sample = self.img
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
        del sample

        gc.collect()
        return result

    def visualize_explanations(self, relevance_imgs, t=None):
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
                hm.save(os.path.join(save_dir, str(i) + '_GuidedBackpropagate_' + words[i] + '.jpg'))
            axes[i].set_title(words[i], fontsize=18)
            axes[i].imshow(hm)
        plt.savefig(os.path.join(save_dir, 'GuidedBackpropagate_hm.jpg'))


class ExplainGridTDGradCam(ExplainGridTDGradient):
    EX_TYPE = 'GradCam'

    def explain_cnn(self, d_img_feature):
        cam_heatmap = self.grad_cam(self.image_features, d_img_feature) #(H, W)
        cam_heatmap = cam_heatmap.unsqueeze(0) #(1, H, W)
        return cam_heatmap

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
        return cam_heatmap.view(-1)

    def visualize_explanations(self, relevance_imgs, t=None):
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
                atten_hm.save(os.path.join(save_dir, str(i) + '_GradCam_' + words[i] + '.jpg'))
            axes[i].set_title(words[i], fontsize=18)
            axes[i].imshow(atten_hm)
        plt.savefig(os.path.join(save_dir, 'GradCAM_hm.jpg'))


class ExplainGridTDGuidedGradCam(ExplainiGridTDGuidedGradient):
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
            image_feature.backward(d_img_feature)
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

    def visualize_explanations(self, relevance_imgs, t=None):
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
                hm.save(os.path.join(save_dir, str(i) + '_GuidedGradCam_' + words[i] + '.jpg'))
            axes[i].set_title(words[i], fontsize=18)
            axes[i].imshow(hm)
        plt.savefig(os.path.join(save_dir, 'GuidedGradCam_hm.jpg'))

'''models with bottomup features'''

class GridTDModelBU(nn.Module):
    '''
    This model add another lstm layer on top of the adaptive attention model
    '''
    EPS = LRPutil.EPSILON
    def __init__(self, embed_dim, hidden_dim, vocab_size, encoder_type):
        super(GridTDModelBU, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.encoder_type = encoder_type
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(0.5)
        # the image encoder to generate image features (bs, C, H, W)
        self.encoder_raw_dim = 2048
        print(f'==========Encoded image feature dim is {self.encoder_raw_dim}==========')
        self.img_projector = nn.Linear(self.encoder_raw_dim, self.hidden_dim)
        self.global_img_feature_proj = nn.Linear(self.hidden_dim, self.embed_dim)
        self.LanguageLSTM = nn.LSTMCell(2*hidden_dim, hidden_dim)
        self.AdaLSTM = AdaptiveLSTMCell(embed_dim*2 + hidden_dim, hidden_dim)
        self.AdaAttention = AdaptiveAttention(self.hidden_dim, 36)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()

    def init_hidden_state(self, V):
        h = torch.zeros(V.shape[0], self.hidden_dim).cuda()
        c = torch.zeros(V.shape[0], self.hidden_dim).cuda()
        return h, c

    def predict_next_word(self,image_feature_proj, xt, states):
        h1t, c1t, h2t, c2t = states  # (bs, hidden_dim, )
        h1t, c1t, st = self.AdaLSTM(xt, (h1t, c1t))  #(bs, hidden_dim, )
        context_t_hat, context_t, alpha_t, beta_t = self.AdaAttention(image_feature_proj, h1t, st)  #(bs, hidden_dim) alpha: (bs, num_pixel), beta:(bs, 1)
        language_input = torch.cat((context_t_hat, h1t), dim=-1) #(bs, 2*hiddendim)
        h2t, c2t = self.LanguageLSTM(language_input, (h2t, c2t))  #(bs, hiddendim)
        predict_score_t = self.fc(self.dropout(context_t_hat + h2t))  # (bs, vocab_size)
        return predict_score_t, alpha_t, beta_t, (h1t, c1t, h2t, c2t)

    def forward(self, images_features, encoded_captions, caption_lengths, ss_prob):
        """
        images: the encoded images from the encoder, of shape (batch_size, C, H, W)
        global_features: the global image features returned by the Encoder, of shape: (batch_size, hidden_dim)
        encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        """
        batch_size = images_features.size(0)
        num_pixels = images_features.size(-2)
        # print(image_features.size(), avg_feature.size())
        image_feature_proj = self.relu(self.img_projector(images_features)) # (bs, num_pixels, hidden_dim)
        # print(image_feature_proj.size())
        avg_feature = torch.mean(image_feature_proj,dim=1) #(bs, hiddendim)
        global_img_feature = self.relu(self.global_img_feature_proj(avg_feature)) #(bs, embedding_dim)
        # print(global_img_feature.size())
        image_feature_proj = image_feature_proj.contiguous()
        image_feature_proj = image_feature_proj.transpose(1,2) # (bs, hidden_dim, num_pixel)
        # print(image_feature_proj.size())
        h1, c1 = self.init_hidden_state(image_feature_proj)
        h2, c2 = self.init_hidden_state(image_feature_proj)
        state = (h1, c1, h2, c2)
        max_length = max(caption_lengths)-1
        predictions = torch.zeros(batch_size, max_length, self.vocab_size).cuda()
        alphas = torch.zeros(batch_size, max_length , num_pixels).cuda()
        betas = torch.zeros(batch_size, max_length,1).cuda()
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
                word_embedding = self.embedding(encoded_captions[:,t])
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            xt = torch.cat((state[2], global_img_feature, word_embedding), dim=-1)   # (batch_size, 2*embed_dim + hidden_dim)
            predict_score_t, alpha_t, beta_t, state = self.predict_next_word(image_feature_proj, xt, state)
            predictions[:, t,:] = predict_score_t
            alphas[:, t, :] = alpha_t
            betas[:, t, :] = beta_t
            last_scores = torch.log_softmax(predict_score_t,-1)
            # print(last_scores.size())
            last_label = torch.argmax(last_scores, -1)  #(batch_size, )
            # print(last_label.size())
        return predictions, alphas, betas, last_scores, max_length

    def sample(self, images_features, word_map, caption_lengths, opt={}):

        batch_size = images_features.size(0)
        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        max_length = max(caption_lengths) - 1

        # print(image_features.size(), avg_feature.size())
        image_feature_proj = self.relu(self.img_projector(images_features))  # (bs, 36, hidden_dim)
        # print(image_feature_proj.size())
        avg_feature = torch.mean(image_feature_proj, dim=1)  # (bs, hiddendim)
        global_img_feature = self.relu(self.global_img_feature_proj(avg_feature))  # (bs, embedding_dim)
        # print(global_img_feature.size())
        image_feature_proj = image_feature_proj.contiguous()
        image_feature_proj = image_feature_proj.transpose(1,2)   # (bs, hidden_dim, num_pixel)
        state = self.init_hidden_state(image_feature_proj) + self.init_hidden_state(image_feature_proj)
        seq = torch.zeros(batch_size,max_length).long().cuda()
        seq_logprobs = torch.zeros(batch_size, max_length).cuda()
        for t in range(max_length):
            if t == 0:
                it = torch.ones(batch_size).long().cuda() * word_map['<start>']
            word_embedding = self.embedding(it)
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            # print(global_img_feature.size(), word_embedding.size(), state[2].size())
            xt = torch.cat((state[2], global_img_feature, word_embedding), dim=-1)  # (batch_size, 2*embed_dim)
            predict_score_t, alpha_t, beta_t, state = self.predict_next_word(image_feature_proj, xt, state)
            predict_score_t = torch.log_softmax(predict_score_t,dim=-1)
            it, sampleLpgprobs = self.sample_next_word(predict_score_t, sample_method, temperature)
            # sample the next word
            if t == 0:
                finished = it == word_map['<end>']
                unfinished = ~finished
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
            bad_sentence= False
            words = sentence.split(' ')
            if len(words) == 0:
                bad_sentence = True
            else:
                while words[-1] in BAD_ENDINGS:
                    words = words[:-1]
                    if len(words) == 0:
                        bad_sentence = True
                        break
            if bad_sentence:
                new_sentences.append(sentence)
            else:
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

            avg_feature = torch.mean(image_feature_proj, dim=1)  # (bs, hiddendim)
            global_img_feature = self.relu(self.global_img_feature_proj(avg_feature))

            image_feature_proj = image_feature_proj.contiguous()
            image_feature_proj = image_feature_proj.transpose(1,2)  # (bs, hidden_dim, num_pixel)
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)   # batch_size, hidden_dim
            image_feature_proj = [image_feature_proj.expand(beam_size, *image_feature_proj.size()[1:]) for g in range(num_group)] # beam_size, hidden_dim, H*W
            global_img_feature = [global_img_feature.expand(beam_size, global_img_feature.size(-1)) for g in range(num_group)] #  beam_size, hidden_dim,
            init_state = self.init_hidden_state(image_feature_proj[0]) + self.init_hidden_state(image_feature_proj[0])
            state = [init_state for g in range(num_group)]  #(ht, ct)
            unfinished_num = [beam_size for g in range(num_group)]
            for step in range(max_cap_length):
                previous_idx = []
                for g in range(num_group):
                    if unfinished_num[g] == 0:
                        continue
                    word_embedding = self.embedding(k_prev_words[g]).squeeze(1)  # unfinished_num, embedding_dim
                    xt = torch.cat((state[g][2], global_img_feature[g], word_embedding), dim=-1)  # (batch_size, 2*embed_dim + hidden_dim)
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
                    beam_idx = top_words // vocab_size  # (unfinished_num, )
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

    def beam_search(self,images_features,  word_map, beam_size=3,max_cap_length=20):
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

            avg_feature = torch.mean(image_feature_proj, dim=1)  # (bs, hiddendim)
            global_img_feature = self.relu(self.global_img_feature_proj(avg_feature))

            image_feature_proj = image_feature_proj.contiguous()
            image_feature_proj = image_feature_proj.transpose(1,2)   # (bs, hidden_dim, num_pixel)
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)   # batch_size, hidden_dim
            # print(global_img_feature.size())
            image_feature_proj = image_feature_proj.expand(beam_size, *image_feature_proj.size()[1:]) # beam_size, hidden_dim, H*W
            global_img_feature = global_img_feature.expand(beam_size, global_img_feature.size(-1)) #  beam_size, hidden_dim,
            state = self.init_hidden_state(image_feature_proj) +  self.init_hidden_state(image_feature_proj)  #(ht, ct)
            unfinished_num = beam_size
            for step in range(max_cap_length):
                word_embedding = self.embedding(k_prev_words).squeeze(1) # unfinished_num, embedding_dim
                xt = torch.cat((state[2], global_img_feature, word_embedding), dim=-1) # (batch_size, 2*embed_dim + hidden_dim)
                predict_score_t, alpha_t, beta_t, state = self.predict_next_word(image_feature_proj, xt, state)
                predict_score_t = torch.log_softmax(predict_score_t,dim=-1) #(unfinished_num, vocab_size)
                top_k_scores_exp = top_k_scores.expand((unfinished_num, vocab_size))
                scores = top_k_scores_exp + predict_score_t
                if step == 0:
                    top_k_scores, top_words = scores[0].topk(beam_size, -1, True, True)  # (unfinished_num, beam_size)
                else:
                    top_k_scores, top_words = scores.view(-1).topk(unfinished_num, -1, True, True) # (unfinished_num, beam_size)
                beam_idx = top_words / vocab_size  # (unfinished_num, )
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
            sen_idx = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<unk>'],word_map['<pad>']}]
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
            image_feature_proj = self.relu(self.img_projector(images_features)) # batch_size, 36, hidden_dim

            avg_feature = torch.mean(image_feature_proj, dim=1)  # (bs, hiddendim)
            global_img_feature = self.relu(self.global_img_feature_proj(avg_feature))

            image_feature_proj = image_feature_proj.contiguous()
            image_feature_proj = image_feature_proj.transpose(1,2)   # (bs, hidden_dim, num_pixel)
            state = self.init_hidden_state(image_feature_proj) +  self.init_hidden_state(image_feature_proj)#(ht, ct)
            for step in range(max_cap_length-1):
                word_embedding = self.embedding(k_prev_words[:, step]) # batch_size, embedding_dim
                if global_img_feature.dim() == 1:
                    global_img_feature = global_img_feature.unsqueeze(0)
                xt = torch.cat((state[2], global_img_feature, word_embedding), dim=-1) # (batch_size, 2*embed_dim + hidden_dim)
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
            return complete_seq, seqs_temp

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

    def get_lrp_weight_step(self, predictions_t, rev_word_map, h2t_, context_hat):
        batch_size, vocab_size = predictions_t.size()
        with torch.no_grad():
            weight_of_context_hat = torch.zeros(batch_size, self.hidden_dim).cuda()
            weight_of_h2t = torch.zeros(batch_size, self.hidden_dim).cuda()
            for b in range(batch_size):
                predicted_labels = torch.argmax(predictions_t[b], dim=-1)  # (the predicted label of image b)  (max_length)
                word_t = predicted_labels.item()
                if rev_word_map[word_t] in STOP_WORDS + ['<start>','<end>','<pad>','<unk>']:
                    continue
                else:
                    word_relevance = torch.zeros(self.vocab_size).cuda()
                    word_relevance[word_t] = predictions_t[b][word_t]
                    r_h2t_context_hat = self.lrp_linear_eps(r_out=word_relevance,
                                                            forward_input=h2t_[b] + context_hat[b],
                                                            forward_output=predictions_t[b],
                                                            weight=self.fc.weight)
                    r_h2t = self.lrp_linear_eps(r_out=r_h2t_context_hat,
                                                      forward_input=h2t_[b],
                                                      forward_output=h2t_[b] + context_hat[b],
                                                      weight=torch.eye(self.hidden_dim).cuda())
                    weight_of_h2t[b] = r_h2t
                    r_context_hat = self.lrp_linear_eps(r_out=r_h2t_context_hat,
                                                        forward_input=context_hat[b],
                                                        forward_output=h2t_[b] + context_hat[b],
                                                        weight=torch.eye(self.hidden_dim).cuda())
                    weight_of_context_hat[b] = r_context_hat
            weight_of_context_hat = LRPutil.normalize_relevance(weight_of_context_hat,dim=-1)
            weight_of_h2t = LRPutil.normalize_relevance(weight_of_h2t, dim=-1)
            return weight_of_context_hat, weight_of_h2t

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
        image_feature_proj = self.relu(self.img_projector(images_features))  # batch_size, 36, hidden_dim

        avg_feature = torch.mean(image_feature_proj, dim=1)  # (bs, hiddendim)
        global_img_feature_before_act = self.global_img_feature_proj(avg_feature)  # (bs, hidden_dim)
        global_img_feature = self.relu(global_img_feature_before_act)
        image_feature_proj = image_feature_proj.contiguous()
        image_feature_proj = image_feature_proj.transpose(1,2)   # (bs, hidden_dim, num_pixel)
        h1, c1 = self.init_hidden_state(image_feature_proj)
        h2, c2 = self.init_hidden_state(image_feature_proj)
        state = (h1, c1, h2, c2)

        max_length = max(caption_lengths) - 1
        predictions = torch.zeros(batch_size, max_length, self.vocab_size).cuda()
        weighted_predictions = torch.zeros(batch_size, max_length, self.vocab_size).cuda()
        for t in range(max_length):
            word_embedding = self.embedding(encoded_captions[:, t])
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            x1t_ = torch.cat((state[2], global_img_feature, word_embedding), dim=-1)  # (batch_size, hidden_dim + embed_dim)
            h1_, c1_, g1_, i1_act_, f1_act_ = lstm_forward(x1t_, state[0], state[1], self.AdaLSTM.lstm_cell.weight_ih,
                                                          self.AdaLSTM.lstm_cell.weight_hh,
                                                          self.AdaLSTM.lstm_cell.bias_ih, self.AdaLSTM.lstm_cell.bias_hh)
            sen_gate = torch.sigmoid(self.AdaLSTM.x_gate(x1t_) + self.AdaLSTM.h_gate(h1_))
            st_ = sen_gate * torch.tanh(c1_)
            context_t_hat_, context_t_, alpha_t_, beta_t_ = self.AdaAttention(image_feature_proj, h1_, st_)  # (1, hidden_dim), (1, num_pixel), (1,1)
            x2t_ = torch.cat((context_t_hat_, h1_), dim=-1)
            h2_, c2_, g2_, i2_act_, f2_act_ = lstm_forward(x2t_, state[2], state[3], self.LanguageLSTM.weight_ih,
                                                          self.LanguageLSTM.weight_hh,
                                                          self.LanguageLSTM.bias_ih, self.LanguageLSTM.bias_hh)
            predict_score_t = self.fc(context_t_hat_ + h2_)  # (bs, vocab_size)
            state = (h1_, c1_, h2_, c2_)
            # print(predict_score_t.size(), x1t_.size(), h1_.size(), c1_.size(), g1_.size(), i1_act_.size(), f1_act_.size(),
            #       alpha_t_.size(), x2t_.size(), h2_.size(), c2_.size(), g2_.size(), i2_act_.size(), f2_act_.size(), context_t_.size(),
            #       context_t_hat_.size(), st_.size(), beta_t_.size())
            predictions[:, t, :] = predict_score_t
            weight_context_hat, weight_h2t = self.get_lrp_weight_step(predict_score_t, rev_word_map, h2_, context_t_hat_)
            weight_prediction_t = self.fc(context_t_hat_*weight_context_hat + weight_h2t * h2_)
            weighted_predictions[:, t, :] = weight_prediction_t
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
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        max_length = max(caption_lengths) - 1
        image_feature_proj = self.relu(self.img_projector(images_features))  # batch_size, 36, hidden_dim

        avg_feature = torch.mean(image_feature_proj, dim=1)  # (bs, hiddendim)
        global_img_feature = self.relu(self.global_img_feature_proj(avg_feature))

        image_feature_proj = image_feature_proj.contiguous()
        image_feature_proj = image_feature_proj.transpose(1,2)  # (bs, hidden_dim, num_pixel)
        state = self.init_hidden_state(image_feature_proj) + self.init_hidden_state(image_feature_proj)
        seq = torch.zeros(batch_size,max_length).long().cuda()
        seq_logprobs = torch.zeros(batch_size, max_length).cuda()

        for t in range(max_length):
            if t == 0:
                it = torch.ones(batch_size).long().cuda() * word_map['<start>']
            word_embedding = self.embedding(it)
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            x1t_ = torch.cat((state[2], global_img_feature, word_embedding), dim=-1)  # (batch_size, hidden_dim + embed_dim)
            h1_, c1_, g1_, i1_act_, f1_act_ = lstm_forward(x1t_, state[0], state[1], self.AdaLSTM.lstm_cell.weight_ih,
                                                          self.AdaLSTM.lstm_cell.weight_hh,
                                                          self.AdaLSTM.lstm_cell.bias_ih, self.AdaLSTM.lstm_cell.bias_hh)
            sen_gate = torch.sigmoid(self.AdaLSTM.x_gate(x1t_) + self.AdaLSTM.h_gate(h1_))
            st_ = sen_gate * torch.tanh(c1_)
            context_t_hat_, context_t_, alpha_t_, beta_t_ = self.AdaAttention(image_feature_proj, h1_, st_)  # (1, hidden_dim), (1, num_pixel), (1,1)
            x2t_ = torch.cat((context_t_hat_, h1_), dim=-1)
            h2_, c2_, g2_, i2_act_, f2_act_ = lstm_forward(x2t_, state[2], state[3], self.LanguageLSTM.weight_ih,
                                                          self.LanguageLSTM.weight_hh,
                                                          self.LanguageLSTM.bias_ih, self.LanguageLSTM.bias_hh)
            state = (h1_, c1_, h2_, c2_)
            predict_score_t = self.fc(context_t_hat_ + h2_)  # (bs, vocab_size)
            weight_context_hat, weight_h2t = self.get_lrp_weight_step(predict_score_t, rev_word_map, h2_, context_t_hat_)
            weight_prediction_t = self.fc(context_t_hat_ * weight_context_hat + weight_h2t * h2_)
            predict_score_t = torch.log_softmax(weight_prediction_t,dim=-1)
            it, sampleLpgprobs = self.sample_next_word(predict_score_t, sample_method, temperature)
            # sample the next word
            if t == 0:
                finished = it == word_map['<end>']
                unfinished = ~finished
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
    parser = config.imgcap_gridTD_argument_parser()
    args = parser.parse_args()
    # for coco
    args.weight = glob.glob('../output/gridTD/vgg16/coco2017/BEST_checkpoint_coco2017_epoch22*')[0]
    args.dataset = 'coco2017'

    #for flickr30k
    # args.weight = glob.glob('../output/gridTD/vgg16/flickr30k/BEST_checkpoint_flickr30k_epoch28*')[0]
    # args.dataset = 'flickr30k'

    word_map_path = f'../dataset/wordmap_{args.dataset}.json'
    word_map = json.load(open(word_map_path, 'r'))
    for explainer in [ExplainGridTDAttention(args, word_map),
                      ExplainGridTDGradient(args, word_map),
                      ExplainiGridTDGuidedGradient(args, word_map),
                      ExplainGridTDGradCam(args, word_map),
                      ExplainGridTDGuidedGradCam(args, word_map)
                      ]:
        explainer.explain_caption(img_filepath,t_list=[3,7])