import os
import torch.nn as nn
import numpy as np
import torch
import glob
from collections import OrderedDict
from pycocoevalcap.bleu.bleu import Bleu as Bleu_scorer
from pycocoevalcap.cider.cider import Cider as CiderD_scorer

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        # print(input.size(), seq.size(), reward.size())
        # print(seq)
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq.detach()>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)
        # print(mask)
        # output = -input * reward
        return output


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, model, optimizer, bleu4, cider, is_best, save_path, encoder):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    if not os.path.isdir(os.path.join(save_path,encoder, data_name)):
        os.makedirs(os.path.join(save_path,encoder, data_name))
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'cider': cider,
             'state_dict': model.state_dict(),
             'optimizer': optimizer}
    filename = f'checkpoint_{data_name}_epoch{epoch}_cider_{cider}.pth'
    torch.save(state, os.path.join(save_path,encoder,data_name,filename))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    # if is_best:
    #     pre_model = glob.glob(os.path.join(save_path, encoder, data_name, f'checkpoint_{data_name}_epoch*'))
    #     for p in pre_model:
    #         os.remove(p)
    #     torch.save(state, os.path.join(save_path, encoder, data_name,'BEST_' +filename))


def adjust_learning_rate(optimizer, shrink_factor, th):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        if param_group['lr'] > th:
            param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    # print(targets)
    mask = targets>0
    scores = scores[mask]
    # print(scores.size())
    targets = targets[mask]
    # print(targets.size())
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.unsqueeze(1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)




def array_to_str(arr, rev_word_map, end_encode):
    ''' when calculating the reward, we prepare the reference and candidate sentence with an <end> symbo so that the model
        learns where to stop. if we do not include the <end>, the model will frequently ends with words such as 'a',
        resulting in a incomplete sentence '''
    out = []
    for i in range(len(arr)):
        if end_encode in out:
            break
        elif rev_word_map[int(arr[i])] not in ['<start>', '<pad>']:
            out.append(rev_word_map[int(arr[i])])
    return ' '.join(out)


def get_self_critical_reward(greedy_res, data_gts, gen_result, word_map, cider_reward_weight,bleu_reward_weight):
    rev_word_map = {v: k for k, v in word_map.items()}
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i], rev_word_map=rev_word_map, end_encode=word_map['<end>'])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i], rev_word_map=rev_word_map,end_encode=word_map['<end>'])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j], rev_word_map=rev_word_map,end_encode=word_map['<end>']) for j in range(len(data_gts[i]))]

    # res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size] for i in range(2 * batch_size)}
    # for i in range(batch_size):
        # print(res__[i])
        # print(res__[i+batch_size])
        # print(gts[i])
    if cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer().compute_score(gts=gts, res=res__)
        # print('Cider scores:', _)
    else:
        cider_scores = 0
    if bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer().compute_score(gts=gts, res=res__)
        bleu_scores = np.array(bleu_scores[3])
        # print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = cider_reward_weight * cider_scores + bleu_reward_weight * bleu_scores

    scores = scores[:batch_size] - scores[batch_size:]
    # print(scores)
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)  # shape(batch_size, seq_length)
    return rewards
