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
import copy
import os
from LRPtools import lrp_wrapper
from LRPtools import utils as LRPutil
import yaml
import skimage.transform

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
        c_t = torch.sum(V * alpha_t, dim=1) #(bs, hidden_dim,)
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
        c_t_hat = beta_t * st + (1-beta_t) * c_t
        return c_t_hat, c_t, alpha_t, beta_t


class AdaptiveAttentionCaptioningModel(nn.Module):

    def __init__(self, embed_dim, hidden_dim, vocab_size, encoder_type):
        super(AdaptiveAttentionCaptioningModel, self).__init__()
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
        self.AdaLSTM = AdaptiveLSTMCell(embed_dim*2, hidden_dim)
        self.AdaAttention = AdaptiveAttention(self.hidden_dim, 196)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()

    def init_hidden_state(self, V):
        h = torch.zeros(V.shape[0], self.hidden_dim).cuda()
        c = torch.zeros(V.shape[0], self.hidden_dim).cuda()
        return h, c

    def predict_next_word(self,image_feature_proj, xt, states):
        ht, ct = states
        ht, ct, st = self.AdaLSTM(xt, (ht, ct))
        # print(ht.size(), ct.size(), st.size())
        context_t_hat, context_t, alpha_t, beta_t = self.AdaAttention(image_feature_proj, ht, st)
        # print(context_t.size(), alpha_t.size(), beta_t.size())
        predict_score_t = self.fc(self.dropout(context_t_hat + ht))  # (bs, vocab_size)
        return predict_score_t, alpha_t, beta_t, (ht, ct)

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

        state = self.init_hidden_state(image_feature_proj)
        max_length = max(caption_lengths)-1
        # print('maxlength', caption_length)
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
            xt = torch.cat((word_embedding, global_img_feature), dim=-1)   # (batch_size, 2*embed_dim)
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
        state = self.init_hidden_state(image_feature_proj)
        seq = torch.zeros(batch_size,max_length).long().cuda()
        seq_logprobs = torch.zeros(batch_size, max_length).cuda()
        for t in range(max_length):
            if t == 0:
                it = torch.ones(batch_size).long().cuda() * word_map['<start>']
            word_embedding = self.embedding(it)
            if global_img_feature.dim() == 1:
                global_img_feature = global_img_feature.unsqueeze(0)
            xt = torch.cat((word_embedding, global_img_feature), dim=-1)  # (batch_size, 2*embed_dim)
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

    def diverse_beam_search(self, imgs, beam_size,word_map, max_cap_length=50, diversity_prob=0.8): # only support batch_size 1
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
            init_state = self.init_hidden_state(image_feature_proj[0])
            state = [init_state for g in range(num_group)]  #(ht, ct)
            unfinished_num = [beam_size for g in range(num_group)]
            for step in range(max_cap_length):
                previous_idx = []
                for g in range(num_group):
                    if unfinished_num[g] == 0:
                        continue
                    word_embedding = self.embedding(k_prev_words[g]).squeeze(1)  # unfinished_num, embedding_dim
                    xt = torch.cat((word_embedding, global_img_feature[g]), dim=-1)  # unfinished_num , 2 * embedding_dim
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
            return return_sentences

    def beam_search(self, imgs, word_map, beam_size=3,max_cap_length=20):
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
            state = self.init_hidden_state(image_feature_proj)  #(ht, ct)
            unfinished_num = beam_size
            for step in range(max_cap_length):
                word_embedding = self.embedding(k_prev_words).squeeze(1) # unfinished_num, embedding_dim
                xt = torch.cat((word_embedding, global_img_feature), dim=-1) # unfinished_num , 2 * embedding_dim
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
            return sentence, sen_idx

    def greedy_search(self, imgs,  word_map, max_cap_length=20):
        self.eval()
        batch_size = imgs.size(0)
        rev_word_map = {v: k for k, v in word_map.items()}
        complete_sentences =[]
        with torch.no_grad():
            k_prev_words = torch.zeros(batch_size, max_cap_length).long().cuda() # (batch_size, caption_length)
            k_prev_words[:, 0] = word_map['<start>'] # the first word is '<start>'
            seqs_temp = [[word_map['<start>']] for _ in range(batch_size)]
            image_features, avg_feature = self.img_encoder(imgs) #
            image_feature_proj = self.relu(self.img_projector(image_features)) # batch_size, hidden_dim, H, W
            image_feature_proj = image_feature_proj.view(batch_size, image_feature_proj.size(1), -1)
            global_img_feature = self.relu(self.global_img_feature_proj(avg_feature))  # batch_size, hidden_dim
            state = self.init_hidden_state(image_feature_proj)  #(ht, ct)
            for step in range(max_cap_length-1):
                word_embedding = self.embedding(k_prev_words[:, step]) # batch_size, embedding_dim
                if global_img_feature.dim() == 1:
                    global_img_feature = global_img_feature.unsqueeze(0)
                xt = torch.cat((word_embedding, global_img_feature), dim=-1) # batch_size , 2 * embedding_dim
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
                complete_sentences.append(sentence)
            return complete_sentences, seqs_temp


class ExplainAdaptiveAttention(object):
    EPS = 0.01
    EX_TYPE = 'lrp'
    def __init__(self, args, word_map, model=None):
        super(ExplainAdaptiveAttention, self).__init__()
        self.args = args
        self.word_map = word_map
        self.vocab_size = len(word_map)
        if model is not None:
            self.model = model
        else:
            self.model = AdaptiveAttentionCaptioningModel(args.embed_dim, args.hidden_dim, len(word_map), args.encoder)
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
        self.xt = torch.zeros(self.caption_length, self.model.embed_dim + self.model.hidden_dim).cuda()
        self.betas = torch.zeros(self.caption_length).cuda()
        self.alphas = torch.zeros(self.caption_length, self.num_pixels).cuda()
        self.ht = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.ct = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.gt = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.it_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.ft_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.st = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context_hat = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        caption = [self.word_map['<start>']]
        for t in range(50):
            it = torch.LongTensor([caption[t]]).cuda()
            word_embedding = self.model.embedding(it)  # (1, embed_dim)
            if self.global_img_feature.dim() == 1:
                self.global_img_feature = self.global_img_feature.unsqueeze(0)
            x_t = torch.cat((word_embedding, self.global_img_feature), dim=-1)   # (1, 2*embed_dim)
            ht_m1 = self.ht[t]
            ct_m1 = self.ct[t]
            h_t, c_t, g_t, i_t_act, f_t_act = self.adalstm_forward(x_t, ht_m1, ct_m1)
            sen_gate = torch.sigmoid(self.model.AdaLSTM.x_gate(x_t) + self.model.AdaLSTM.h_gate(ht_m1))
            s_t = sen_gate * torch.tanh(c_t)
            context_t_hat, context_t, alpha_t, beta_t = self.model.AdaAttention(self.image_feature_proj, h_t.unsqueeze(0), s_t) #(1, hidden_dim), (1, num_pixel), (1,1)
            predict_score_t = self.model.fc(context_t_hat + h_t) #(1, vocab_size)
            label = torch.argmax(predict_score_t)
            if label == self.word_map['<end>']:
                print(caption)
                return
            caption.append(label)
            # here we save the intermediate states for further relevance backpropagation
            self.xt[t] = x_t[0]
            self.predictions[t,:] = predict_score_t[0]
            self.alphas[t] = alpha_t[0]
            self.betas[t] = beta_t[0]
            self.ht[t+1] = h_t
            self.ct[t+1] = c_t
            self.gt[t] = g_t
            self.it_act[t] = i_t_act
            self.ft_act[t] = f_t_act
            self.st[t] = s_t[0]
            self.context[t] = context_t[0]
            self.context_hat[t] = context_t_hat[0]

    def get_hidden_parameters(self, img_filepath ):
        self.img = self.preprocess_img(img_filepath) #(bs, C, H, W)
        self.beam_caption, self.beam_caption_encode = self.model.beam_search(self.img, self.word_map, beam_size=3, max_cap_length=20)
        self.beam_caption_encode = [self.word_map['<start>']] + self.beam_caption_encode
        print(f'the predicted caption of {img_filepath} is "{self.beam_caption[0]}"')
        # perform the forward pass and save the intermediate variables
        self.image_features, self.avg_feature = self.model.img_encoder(self.img)  # (bs, fea_dim, H, W), (bs, fea_dim)
        self.num_pixels = self.image_features.size(-1) * self.image_features.size(-2)
        self.image_feature_proj = self.model.relu(self.model.img_projector(self.image_features))  # (bs, hiddendim, H, W)
        self.global_img_feature = self.model.relu(self.model.global_img_feature_proj(self.avg_feature))  # (bs, embedding_dim)
        self.image_feature_proj = self.image_feature_proj.contiguous()
        self.image_feature_proj = self.image_feature_proj.view(1, self.model.hidden_dim, -1)  # (bs, hidden_dim, num_pixel)
        self.caption_length = len(self.beam_caption_encode) - 1
        self.predictions = torch.zeros(self.caption_length, self.vocab_size).cuda()
        self.xt = torch.zeros(self.caption_length, self.model.embed_dim + self.model.hidden_dim).cuda()
        self.betas = torch.zeros(self.caption_length).cuda()
        self.alphas = torch.zeros(self.caption_length, self.num_pixels).cuda()
        self.ht = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.ct = torch.zeros(self.caption_length + 1, self.model.hidden_dim).cuda()
        self.gt = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.it_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.ft_act = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.st = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context_hat = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        for t in range(self.caption_length):
            it = torch.LongTensor([self.beam_caption_encode[t]]).cuda()
            word_embedding = self.model.embedding(it)  # (1, embed_dim)
            if self.global_img_feature.dim() == 1:
                self.global_img_feature = self.global_img_feature.unsqueeze(0)
            x_t = torch.cat((word_embedding, self.global_img_feature), dim=-1)   # (1, 2*embed_dim)
            ht_m1 = self.ht[t]
            ct_m1 = self.ct[t]
            h_t, c_t, g_t, i_t_act, f_t_act = self.adalstm_forward(x_t, ht_m1, ct_m1)
            sen_gate = torch.sigmoid(self.model.AdaLSTM.x_gate(x_t) + self.model.AdaLSTM.h_gate(ht_m1))
            s_t = sen_gate * torch.tanh(c_t)
            context_t_hat, context_t, alpha_t, beta_t = self.model.AdaAttention(self.image_feature_proj, h_t.unsqueeze(0), s_t) #(1, hidden_dim), (1, num_pixel), (1,1)
            predict_score_t = self.model.fc(context_t_hat + h_t) #(1, vocab_size)
            # here we save the intermediate states for further relevance backpropagation
            # print(x_t.size(), predict_score_t.size(), alpha_t.size(), beta_t.size(), h_t.size(), c_t.size(), g_t.size(), i_t_act.size(), f_t_act.size(),s_t.size(),context_t.size(),context_t_hat.size())
            self.xt[t] = x_t[0]
            self.predictions[t,:] = predict_score_t[0]
            self.alphas[t] = alpha_t[0]
            self.betas[t] = beta_t[0]
            self.ht[t+1] = h_t
            self.ct[t+1] = c_t
            self.gt[t] = g_t
            self.it_act[t] = i_t_act
            self.ft_act[t] = f_t_act
            self.st[t] = s_t[0]
            self.context[t] = context_t[0]
            self.context_hat[t] = context_t_hat[0]

    def explain_caption_wordt(self, t):
        assert t < self.caption_length  #(t starts from 0)
        preceeding_cap_length = t+1
        target_word_encode = self.beam_caption_encode[t+1]
        words = self.beam_caption[0].split(' ')
        print(target_word_encode, words[t])
        weight_ig = self.adalstm_weight_i.chunk(4,0)[2]  #(hidden_dim, embed_dim + hidden_dim)
        weight_hg = self.adalstm_weight_h.chunk(4,0)[2]  #(hidden_dim, hidden_dim)
        weight_g = torch.cat((weight_ig, weight_hg), dim=1) #(hidden_dim, 2 * hidden_dim+embed_dim)
        xht = torch.cat((self.xt[:preceeding_cap_length], self.ht[:preceeding_cap_length]), dim=1) #(preceeding_length, 2*hidden_dim+embed_dim)
        predict_score_t = self.predictions[t] #(vocat_size,)
        word_relevance = torch.zeros(1, self.vocab_size).cuda()
        word_relevance[0, target_word_encode] = predict_score_t[target_word_encode]
        self.r_ht = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        self.r_ct = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        # r_gt = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        self.r_xht = torch.zeros(preceeding_cap_length, self.model.hidden_dim+ 2 * self.model.embed_dim).cuda()
        self.r_global_img_feature = torch.zeros(self.model.embed_dim).cuda()
        self.r_word_embedding = torch.zeros(preceeding_cap_length, self.model.embed_dim).cuda()
        self.r_img_feature = torch.zeros(self.num_pixels, self.model.encoder_raw_dim).cuda()
        self.r_img_feature_proj = torch.zeros(self.num_pixels, self.model.hidden_dim).cuda()
        r_ht_context = self.lrp_linear_eps(r_out=word_relevance,
                                           forward_input=self.ht[t+1]+self.context_hat[t],
                                           forward_output=predict_score_t,
                                           weight=self.output_weight)
        # print('r_ht_context',r_ht_context.size())
        self.r_ht[t+1] = self.lrp_linear_eps(r_out=r_ht_context,
                                             forward_input=self.ht[t+1],
                                             forward_output=self.ht[t+1]+self.context_hat[t],
                                             weight=torch.eye(self.model.hidden_dim).cuda())

        r_context_hat = self.lrp_linear_eps(r_out=r_ht_context,
                                            forward_input=self.context_hat[t],
                                            forward_output=self.ht[t+1]+self.context_hat[t],
                                            weight=torch.eye(self.model.hidden_dim).cuda())
        r_context = self.lrp_linear_eps(r_out=r_context_hat,
                                        forward_input=(1-self.betas[t])*self.context[t],
                                        forward_output=self.context_hat[t],
                                        weight=torch.eye(self.model.hidden_dim).cuda())
        # print('r_context',r_context.size())
        r_st = self.lrp_linear_eps(r_out=r_context_hat,
                                   forward_input=self.betas[t]*self.st[t],
                                   forward_output=self.context_hat[t],
                                   weight=torch.eye(self.model.hidden_dim).cuda())
        # print('r_st', r_st.size())
        self.r_ct[t+1] = r_st
        for i in range(preceeding_cap_length)[::-1]:
            self.r_ct[i+1] = self.r_ct[i+1] + self.r_ht[i+1]
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
                                                forward_output=torch.tanh(self.gt[i]),
                                                weight=weight_g)
            self.r_ht[i] = self.r_xht[i][self.model.embed_dim*2:]
            if i == t:
                self.r_global_img_feature = self.r_global_img_feature + self.r_xht[i][self.model.embed_dim:self.model.embed_dim*2]
            self.r_word_embedding[i] = self.r_xht[i][:self.model.embed_dim]
        r_average_img_feature = self.lrp_linear_eps(r_out=self.r_global_img_feature,
                                                    forward_input=self.avg_feature,
                                                    forward_output=False,
                                                    weight=self.model.global_img_feature_proj.weight)
        # print(r_average_img_feature.size())
        image_feature = self.image_features.view(1, self.model.encoder_raw_dim, self.num_pixels)
        image_feature = image_feature.transpose(1,2).squeeze(0) #(num_pixel, encode_raw_dim)
        image_feature_proj = self.image_feature_proj.transpose(1,2).squeeze(0) #(num_pixel, hidden_dim)
        for i in range(self.num_pixels):
            self.r_img_feature[i] = self.lrp_linear_eps(r_out=r_average_img_feature,
                                                        forward_input=image_feature[i]/self.num_pixels,
                                                        forward_output=self.avg_feature,
                                                        weight=torch.eye(self.model.encoder_raw_dim).cuda())
            self.r_img_feature_proj[i] = self.lrp_linear_eps(r_out=r_context,
                                                             forward_input=image_feature_proj[i] * self.alphas[t][i],
                                                             forward_output=self.context[t],
                                                             weight=torch.eye(self.model.hidden_dim).cuda())
            self.r_img_feature[i] = self.r_img_feature[i] + self.lrp_linear_eps(r_out=self.r_img_feature_proj[i],
                                                                                forward_input=image_feature[i],
                                                                                forward_output=False,
                                                                                weight=self.model.img_projector.weight.squeeze(-1).squeeze(-1))
        r_words = torch.sum(self.r_word_embedding, dim=-1)
        max_abs_r_words = torch.max(torch.abs(r_words))
        if max_abs_r_words > 0:
            r_words = r_words / max_abs_r_words
        r_img_feature = self.r_img_feature.unsqueeze(0).transpose(1,2).view(self.image_features.size())
        # print(torch.sum(r_img_feature>0), torch.sum(r_img_feature==0), torch.sum(r_img_feature<0))
        torch.cuda.empty_cache()
        return r_img_feature, r_words

    def explain_cnn(self, r_img_feature):
        # cnn_encoder = copy.deepcopy(self.model.img_encoder.encoder)
        relevance_img = self.model.img_encoder.encoder.compute_lrp(self.img, target=r_img_feature)
        print(torch.sum(relevance_img > 0), torch.sum(relevance_img == 0), torch.sum(relevance_img < 0))
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
            hm_show = Image.blend(img_original, hm, 0.8)
            if isinstance(t, list) and  i in t:
                hm_show.save(os.path.join(save_dir, str(i) + '_lrp_' + words[i] + '.jpg'))
            axes[i].set_title(words[i], fontsize=18)
            axes[i].imshow(hm)
        plt.savefig(os.path.join(save_dir, 'lrp_hm.jpg'))
        _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20, 20))
        axes = axes.flatten()
        for i in range(self.caption_length):
            atten_hm = LRPutil.visuallize_attention(img_original, self.alphas[i],
                                                    (int(np.sqrt(self.num_pixels)),int(np.sqrt(self.num_pixels))), upscale=16)
            if isinstance(t, list) and  i in t:
                atten_hm.save(os.path.join(save_dir, str(i) + '_attention_' + words[i] + '.jpg'))
            axes[i].set_title(words[i], fontsize=18)
            axes[i].imshow(atten_hm)
        plt.savefig(os.path.join(save_dir, 'attention_hm.jpg'))


class ExplainAdaptiveGradient(object):
    EX_TYPE = 'gradient'
    def __init__(self, args, word_map):
        super(ExplainAdaptiveGradient, self).__init__()
        self.args = args
        self.word_map = word_map
        self.vocab_size = len(word_map)
        self.model = AdaptiveAttentionCaptioningModel(args.embed_dim, args.hidden_dim, len(word_map), args.encoder)
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

    def preprocess_img(self, img_filepath):
        image_data = Image.open(img_filepath).convert('RGB')
        img = self.img_transform(image_data)
        img = img.unsqueeze(0).cuda()
        return img

    def get_hidden_parameters(self, img_filepath):
        self.img = self.preprocess_img(img_filepath)  # (bs, C, H, W)
        self.beam_caption, self.beam_caption_encode = self.model.beam_search(self.img, self.word_map, beam_size=3,
                                                                             max_cap_length=20)
        self.beam_caption_encode = [self.word_map['<start>']] + self.beam_caption_encode
        print(f'the predicted caption of {img_filepath} is "{self.beam_caption[0]}"')
        # perform the forward pass and save the intermediate variables
        self.image_features, self.avg_feature = self.model.img_encoder(self.img)  # (bs, fea_dim, H, W), (bs, fea_dim)
        self.num_pixels = self.image_features.size(-1) * self.image_features.size(-2)
        self.image_feature_proj = self.model.relu(self.model.img_projector(self.image_features))  # (bs, hiddendim, H, W)
        self.global_img_feature = self.model.relu(self.model.global_img_feature_proj(self.avg_feature))  # (bs, embedding_dim)
        self.image_feature_proj = self.image_feature_proj.contiguous()
        self.image_feature_proj = self.image_feature_proj.view(1, self.model.hidden_dim, -1)  # (bs, hidden_dim, num_pixel)
        self.caption_length = len(self.beam_caption_encode) - 1
        self.predictions = torch.zeros(self.caption_length, self.vocab_size).cuda()
        self.xt = torch.zeros(self.caption_length, self.model.embed_dim + self.model.hidden_dim).cuda()
        self.betas = torch.zeros(self.caption_length).cuda()
        self.alphas = torch.zeros(self.caption_length, self.num_pixels).cuda()
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
        self.st = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context_hat = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.context = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        self.sen_gate = torch.zeros(self.caption_length, self.model.hidden_dim).cuda()
        for t in range(self.caption_length):
            it = torch.LongTensor([self.beam_caption_encode[t]]).cuda()
            word_embedding = self.model.embedding(it)  # (1, embed_dim)
            if self.global_img_feature.dim() == 1:
                self.global_img_feature = self.global_img_feature.unsqueeze(0)
            x_t = torch.cat((word_embedding, self.global_img_feature), dim=-1)  # (1, 2*embed_dim)
            ht_m1 = self.ht[t]
            ct_m1 = self.ct[t]
            h_t, c_t, i_t, f_t, g_t, o_t, i_t_act, f_t_act, g_t_act, o_t_act = self.adalstm_forward(x_t, ht_m1, ct_m1)
            sen_gate_t = torch.sigmoid(self.model.AdaLSTM.x_gate(x_t) + self.model.AdaLSTM.h_gate(ht_m1))
            s_t = sen_gate_t * torch.tanh(c_t)
            context_t_hat, context_t, alpha_t, beta_t = self.model.AdaAttention(self.image_feature_proj, h_t.unsqueeze(0), s_t)  # (1, hidden_dim), (1, num_pixel), (1,1)
            predict_score_t = self.model.fc(context_t_hat + h_t)  # (1, vocab_size)
            # here we save the intermediate states for further relevance backpropagation
            self.sen_gate[t] = sen_gate_t[0]
            self.xt[t] = x_t[0]
            self.predictions[t, :] = predict_score_t[0]
            self.alphas[t] = alpha_t[0]
            self.betas[t] = beta_t[0]
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
            self.st[t] = s_t[0]
            self.context[t] = context_t[0]
            self.context_hat[t] = context_t_hat[0]

    def explain_caption_wordt(self,t):
        assert t < self.caption_length  #(t starts from 0)
        preceeding_cap_length = t+1
        target_word_encode = self.beam_caption_encode[t+1]
        d_word_pred = torch.zeros(1, self.vocab_size).cuda()
        d_word_pred[0, target_word_encode] = 1  #(1, vocab_size)
        d_ht = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        d_ct = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        d_it = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_ft = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_gt = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_ot = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_it_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_ft_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_gt_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_ot_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_xt = torch.zeros(preceeding_cap_length, 2* self.model.embed_dim).cuda()
        d_global_img_feature = torch.zeros(self.model.embed_dim).cuda()
        d_word_embedding = torch.zeros(preceeding_cap_length, self.model.embed_dim).cuda()
        d_img_feature = torch.zeros(self.num_pixels, self.model.encoder_raw_dim).cuda()
        d_img_feature_proj = torch.zeros(self.num_pixels, self.model.hidden_dim).cuda()
        #backward starts
        d_ht_context = torch.matmul(d_word_pred, self.output_weight).squeeze()
        d_c_hat = d_ht_context * 1
        d_context = d_c_hat * (1-self.betas[t])
        d_st = d_c_hat * self.betas[t]
        d_ct[t+1] += d_st * self.sen_gate[t] * (1-(torch.tanh(self.ct[t+1]))**2)
        d_ht[t+1] = d_ht_context * 1
        for i in range(self.num_pixels):
            d_img_feature_proj[i] = d_context * self.alphas[t, i]
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
            d_ht[i] = torch.matmul(d_gates, self.adalstm_weight_h).squeeze() #(hidden_dim)
            d_xt[i] = torch.matmul(d_gates, self.adalstm_weight_i).squeeze() #(2*embed_dim)
            d_global_img_feature = d_global_img_feature + d_xt[i][self.model.embed_dim:] #(embed_dim, )
            d_word_embedding[i] = d_xt[i][:self.model.embed_dim]
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

    def explain_caption(self, img_filepath,t_list=None):
        self.img_filepath = img_filepath
        self.get_hidden_parameters(img_filepath)
        self.image_feature_proj = self.image_feature_proj.transpose(1,2)
        relevance_imgs = []
        relevance_preceeding_words = []
        for t in range(self.caption_length):
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
            if isinstance(t, list) and  i in t:
                hm = Image.fromarray(np.uint8(hm * 255))
                hm.save(os.path.join(save_dir, str(i) + '_gradient_' + words[i] + '.jpg'))
            axes[i].set_title(words[i], fontsize=18)
            axes[i].imshow(hm)
        plt.savefig(os.path.join(save_dir,'gradient_hm.jpg'))


class ExplainiAdaptiveGuidedGradient(ExplainAdaptiveGradient):
    EX_TYPE = 'GuidedBackpropagate'
    def explain_caption_wordt(self,t):
        assert t < self.caption_length  #(t starts from 0)
        preceeding_cap_length = t+1
        target_word_encode = self.beam_caption_encode[t+1]
        d_word_pred = torch.zeros(1, self.vocab_size).cuda()
        d_word_pred[0, target_word_encode] = 1  #(1, vocab_size)
        d_ht = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        d_ct = torch.zeros(preceeding_cap_length+1, self.model.hidden_dim).cuda()
        d_it = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_ft = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_gt = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_ot = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_it_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_ft_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_gt_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_ot_act = torch.zeros(preceeding_cap_length, self.model.hidden_dim).cuda()
        d_xt = torch.zeros(preceeding_cap_length, self.model.hidden_dim + self.model.embed_dim).cuda()
        d_global_img_feature = torch.zeros(self.model.embed_dim).cuda()
        d_word_embedding = torch.zeros(preceeding_cap_length, self.model.embed_dim).cuda()
        d_img_feature = torch.zeros(self.num_pixels, self.model.encoder_raw_dim).cuda()
        d_img_feature_proj = torch.zeros(self.num_pixels, self.model.hidden_dim).cuda()
        #backward starts
        d_ht_context = torch.matmul(d_word_pred, self.output_weight).squeeze()
        d_c_hat = d_ht_context * 1
        d_st = d_c_hat * self.betas[t]
        d_ct[t+1] += d_st * self.sen_gate[t] * (1-(torch.tanh(self.ct[t+1]))**2)
        d_context = d_c_hat * (1-self.betas[t])
        d_ht[t+1] = d_ht_context * 1
        for i in range(self.num_pixels):
            d_img_feature_proj[i] = d_context * self.alphas[t, i]
        d_img_feature_proj[self.image_feature_proj[0]<0] = 0
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
            d_ht[i] = torch.matmul(d_gates, self.adalstm_weight_h).squeeze() #(hidden_dim)
            d_xt[i] = torch.matmul(d_gates, self.adalstm_weight_i).squeeze() #(2*embed_dim)
            d_global_img_feature = d_global_img_feature + d_xt[i][self.model.embed_dim:] #(embed_dim, )
            d_word_embedding[i] = d_xt[i][:self.model.embed_dim]
        d_average_img_feature = torch.matmul(d_global_img_feature, self.model.global_img_feature_proj.weight).squeeze()
        d_global_img_feature[self.global_img_feature[0]<0] = 0
        for i in range(self.num_pixels):
            d_img_feature[i] = 1.0 * d_average_img_feature / self.num_pixels
            d_img_feature[i] = d_img_feature[i] + torch.matmul(d_img_feature_proj[i], self.model.img_projector.weight.squeeze(-1).squeeze(-1)).squeeze()
        r_words = torch.sum(d_word_embedding, dim=-1)
        max_abs_r_words = torch.max(torch.abs(r_words))
        if max_abs_r_words > 0:
            r_words = r_words / max_abs_r_words
        d_img_feature = d_img_feature.unsqueeze(0).transpose(1,2).view(self.image_features.size())
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

    def explain_cnn(self, d_img_feature):
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
        del cnn_encoder
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


class ExplainAdaptiveGradCam(ExplainAdaptiveGradient):
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


class ExplainAdaptiveGuidedGradCam(ExplainiAdaptiveGuidedGradient):
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
        cam = skimage.transform.pyramid_expand(cam.detach().cpu().numpy(), upscale=32,multichannel=False)
        guided_results = guided_gradient * torch.from_numpy(cam).cuda().float().expand_as(guided_gradient)
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


if __name__ == '__main__':
    import config
    import json
    img_filepath = '/home/sunjiamei/work/ImageCaptioning/dataset/flickr30k/Flickr30k_Dataset/1009434119.jpg'
    parser = config.imgcap_adaptive_argument_parser()
    args = parser.parse_args()
    word_map_path = f'../dataset/wordmap_{args.dataset}.json'
    word_map = json.load(open(word_map_path, 'r'))
    explainer = ExplainAdaptiveAttention(args, word_map)
    # explainer = ExplainAdaptiveGradient(args, word_map)
    # explainer = ExplainiAdaptiveGuidedGradient(args, word_map)
    # explainer = ExplainAdaptiveGradCam(args, word_map)
    # explainer = ExplainAdaptiveGuidedGradCam(args, word_map)
    # explainer.forward_greedy(img_filepath)
    explainer.explain_caption(img_filepath)