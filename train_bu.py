import json
import torch
from config import imgcap_adaptive_argument_parser, imgcap_gridTD_argument_parser, imgcap_aoa_argument_parser
import torchvision.transforms as transforms
from dataset.dataloader import ImagecapDataset, ImagecapDatasetFromFeature
from models import adaptiveattention
from models import gridTDmodel
from models import aoamodel
import models.modelutils as mutils
from models.metrics import BLEU, CIDEr, SPICE, ROUGE
import os
import glob

def main(args):
    print(f'The arguments are')
    print(args)
    print(f'model_type is {args.model_type}')
    word_map_path = f'./dataset/wordmap_{args.dataset}.json'
    word_map = json.load(open(word_map_path, 'r'))

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    print('==========Loading Data==========')
    train_data = ImagecapDatasetFromFeature(args.dataset, 'train', train_transform, )
    val_data = ImagecapDatasetFromFeature(args.dataset, 'val', val_transform, )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, pin_memory=True, sampler=None)
    print(len(train_loader))

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.workers,
                                             pin_memory=True)
    print(len(val_loader))

    print('==========Data Loaded==========')
    print('==========Setting Model==========')
    if args.model_type == 'adaptive':
        model = adaptiveattention.AdaptiveAttentionCaptioningModel(args.embed_dim, args.hidden_dim, len(word_map), args.encoder)
    elif args.model_type == 'gridtd':
        model = gridTDmodel.GridTDModelBU(args.embed_dim, args.hidden_dim, len(word_map), args.encoder)
    elif args.model_type == 'aoa':
        model = aoamodel.AOAModelBU(args.embed_dim, args.hidden_dim, args.num_head, len(word_map), args.encoder)
    else:
        raise NotImplementedError(f'model_type {args.model_type} does not available yet')
    model.cuda()

    if args.resume:
        print(f'==========Resuming weights from {args.resume}==========')
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_cider = checkpoint['cider']
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print(f'==========Initializing model from random==========')
        start_epoch = 0
        epochs_since_improvement = 0
        best_cider = 0
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.decoder_lr,
                                 betas=(0.8, 0.999))

    print(f'==========Start Training==========')
    for epoch in range(start_epoch, args.epochs):
        # if args.model_type == 'aoa':
        #     if epoch > 0 and (epoch)%3==0:
        #         mutils.adjust_learning_rate(optimizer, 0.8, 2e-5)
        if epochs_since_improvement >=2:
            mutils.adjust_learning_rate(optimizer, 0.8, 2e-5)
            epochs_since_improvement = 0
        if args.cider_tune:
            print(f'==========Training with Cider Optm==========')
            criterion = mutils.RewardCriterion().cuda()
            train_func = traincider
        elif args.lrp_tune:
            print(f'==========Training with lrp Optm==========')
            criterion = torch.nn.CrossEntropyLoss(ignore_index=word_map['<pad>']).cuda()
            train_func = train_lrp
        elif args.lrp_cider_tune:
            print(f'==========Training with lrp cider Optm==========')
            criterion = mutils.RewardCriterion().cuda()
            train_func = trainciderlrp
        else:
            print(f'==========Training ==========')
            criterion = torch.nn.CrossEntropyLoss(ignore_index=word_map['<pad>']).cuda()
            train_func = train
            # args.ss_prob = args.ss_prob + (epoch //10) * 0.03
            # print(f'Traning with ss_prob {args.ss_prob}')
        train_func(train_loader, model, criterion, optimizer, epoch, args.ss_prob, word_map, args.print_freq, args.grad_clip)

        bleu, cider = validate(val_loader,model, word_map, 3, epoch, beam_search_type='beam_search')
        is_best = cider > best_cider
        best_cider = max(cider, best_cider)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement))
        else:
            epochs_since_improvement = 0
        if args.lrp_tune:
            mutils.save_checkpoint(args.dataset, str(epoch)+'lrp', epochs_since_improvement, model, optimizer,bleu,cider,is_best, args.save_path, args.encoder)
        else:
            mutils.save_checkpoint(args.dataset, epoch, epochs_since_improvement, model, optimizer,bleu,cider,is_best, args.save_path, args.encoder)


def train(train_loader, model, criterion, optimizer, epoch, ss_prob, word_map, print_freq, grad_clip):
    model.train()
    losses = mutils.AverageMeter()         # loss (per decoded word)
    top5accs = mutils.AverageMeter()       # top5 accuracy

    for i, (imgs, caps, all_caps, caplens) in enumerate(train_loader):
        imgs = imgs.cuda()
        caps = caps.cuda()
        predictions, alphas, betas, _, max_length = model(imgs, caps, caplens, ss_prob)
        targets = caps[:, 1:max_length+1]
        scores = predictions.contiguous().view(-1, predictions.size(2))
        targets = targets.contiguous().view(predictions.size(0)* predictions.size(1))
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            mutils.clip_gradient(optimizer,grad_clip=grad_clip)
        optimizer.step()
        top5 = mutils.accuracy(scores, targets, 1)
        losses.update(loss.item(), sum(caplens).float())
        top5accs.update(top5, sum(caplens).float())
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-1 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(epoch, i, len(train_loader),
                                                                            loss=losses,
                                                                            top5=top5accs))


def traincider(train_loader, model, criterion, optimizer, epoch, ss_prob, word_map, print_freq, grad_clip):
    model.train()
    losses = mutils.AverageMeter()  # loss (per decoded word)
    rewards = mutils.AverageMeter()
    for i, (imgs, caps, all_caps, caplens) in enumerate(train_loader):
        imgs = imgs.cuda()
        model.eval()
        with torch.no_grad():
            greedy_res , _, _ = model.sample(imgs, word_map, caplens)
        model.train()
        gen_result, sample_logprobs, max_length = model.sample(imgs, word_map, caplens, opt={'sample_method':'sample'})
        reward = mutils.get_self_critical_reward(greedy_res, all_caps, gen_result, word_map, cider_reward_weight=1., bleu_reward_weight=0)
        reward = torch.from_numpy(reward).float().cuda()
        loss = criterion(sample_logprobs, gen_result.data, reward)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            mutils.clip_gradient(optimizer,grad_clip=grad_clip)
        optimizer.step()
        losses.update(loss.item(), sum(caplens-2).float())
        rewards.update(reward[:,0].mean().item(), float(len(reward)))
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Reward {rewards.val:.3f} ({rewards.avg:.3f})\t'.format(epoch, i, len(train_loader),
                                                                            loss=losses,
                                                                            rewards=rewards))


def train_lrp(train_loader, model, criterion, optimizer, epoch, ss_prob, word_map, print_freq, grad_clip):
    model.train()
    losses = mutils.AverageMeter()         # loss (per decoded word)
    top5accs = mutils.AverageMeter()       # top5 accuracy
    rev_word_map = {v: k for k, v in word_map.items()}
    for i, (imgs, caps, all_caps, caplens) in enumerate(train_loader):
        # print(imgs.size())
        imgs = imgs.cuda()
        caps = caps.cuda()
        predictions, weighted_predictions, max_length = model.forwardlrp_context(imgs,caps, caplens, rev_word_map)
        scores = predictions.contiguous().view(-1, predictions.size(2))
        targets = caps[:, 1:max_length + 1]
        targets = targets.contiguous().view(predictions.size(0) * predictions.size(1))
        loss_standard = criterion(scores, targets)

        # print(weighted_predictions.size(), max_length)
        weighted_scores = weighted_predictions.contiguous().view(-1, weighted_predictions.size(2))
        loss_lrp = criterion(weighted_scores, targets)
        loss = loss_lrp + loss_standard
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            mutils.clip_gradient(optimizer,grad_clip=grad_clip)
        optimizer.step()
        top5 = mutils.accuracy(scores, targets,1)
        losses.update(loss.item(), sum(caplens).float())
        top5accs.update(top5, sum(caplens).float())
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-1 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(epoch, i, len(train_loader),
                                                                            loss=losses,
                                                                            top5=top5accs))
            # state = {'epoch': epoch,
            #          'acctop1': top5accs,
            #          'loss': losses,
            #          'state_dict': model.state_dict(),
            #          'batch': i}
            # filename = f'lrp_checkpoint_epoch{epoch}_batch_{i}.pth'
            # torch.save(state, os.path.join('/home/sunjiamei/work/ImageCaptioning/ImgCaptioningPytorch/output/gridTD/vgg16/flickr30k/lrpfinetune/', filename))


def trainciderlrp(train_loader, model, criterion, optimizer, epoch, ss_prob, word_map, print_freq, grad_clip):
    model.train()
    losses = mutils.AverageMeter()  # loss (per decoded word)
    rewards = mutils.AverageMeter()
    rev_word_map = {v: k for k, v in word_map.items()}
    for i, (imgs, caps, all_caps, caplens) in enumerate(train_loader):
        imgs = imgs.cuda()
        model.eval()
        with torch.no_grad():
            greedy_res , _, _ = model.sample(imgs, word_map, caplens)
        model.train()
        gen_result, sample_logprobs, max_length = model.sample_lrp(imgs, rev_word_map, word_map, caplens, opt={'sample_method':'sample'})
        reward = mutils.get_self_critical_reward(greedy_res, all_caps, gen_result, word_map, cider_reward_weight=1., bleu_reward_weight=0)
        reward = torch.from_numpy(reward).float().cuda()
        loss = criterion(sample_logprobs, gen_result.data, reward)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            mutils.clip_gradient(optimizer,grad_clip=grad_clip)
        optimizer.step()
        losses.update(loss.item(), sum(caplens-2).float())
        rewards.update(reward[:,0].mean().item(), float(len(reward)))
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Reward {rewards.val:.3f} ({rewards.avg:.3f})\t'.format(epoch, i, len(train_loader),
                                                                            loss=losses,
                                                                            rewards=rewards))

def validate(val_loader, model, word_map, beam_size, epoch, beam_search_type='greedy'):
    model.eval()
    rev_word_map = {v: k for k, v in word_map.items()}
    with torch.no_grad():
        references = {}  # references (true captions) for calculating BLEU-4 score
        hypotheses = {}  # hypotheses (predictions)
        prediction_save = {} # because each image may have multiple predictions, we use another dict to save all the captions for one images with the key as filename
        gt_save = {}
        image_id = 0
        for i, (imgs, allcaps, caplens, img_filenames) in enumerate(val_loader):
            imgs = imgs.cuda()
            if beam_search_type == 'dbs':
                sentences = model.diverse_beam_search(imgs,  beam_size, word_map)
            elif beam_search_type == 'beam_search':
                sentences, _ = model.beam_search(imgs,  word_map, beam_size=beam_size)
            elif beam_search_type == 'greedy':
                sentences, _ = model.greedy_search(imgs,  word_map)
            else:
                raise NotImplementedError(
                    'please specify the decoding method in [dbs, beam_search, greedy] in string type')
            # assert len(sentences) == batch_size
            img_filename = img_filenames[0]
            if img_filename not in prediction_save.keys():
                prediction_save[img_filename] = []
                gt_save[img_filename] = []
            for idx , sentence in enumerate(sentences):
                if not image_id in hypotheses.keys():
                    hypotheses[image_id] = []
                    references[image_id] = []
                hypotheses[image_id].append({'caption':sentence})
                prediction_save[img_filename].append(sentence)
                for ref_item in allcaps[0]:
                    # print(ref_item)
                    enc_ref = [w.item() for w in ref_item if w.item() not in {word_map['<start>'], word_map['<end>'], word_map['<pad>'], word_map['<unk>']}]
                    ref = ' '.join([rev_word_map[enc_ref[i]] for i in range(len(enc_ref))])
                    if ref not in gt_save[img_filename]:
                        gt_save[img_filename].append(ref)
                    references[image_id].append({'caption':ref})
                image_id += 1
    # print(hypotheses)
    print("Calculating Evalaution Metric Scores......\n")
    avg_bleu_dict = BLEU().calculate(hypotheses,references)
    bleu4 = avg_bleu_dict['bleu_4']
    avg_cider_dict = CIDEr().calculate(hypotheses, references)
    cider = avg_cider_dict['cider']
    avg_spice_dict = SPICE().calculate(hypotheses, references)
    avg_rouge_dict = ROUGE().calculate(hypotheses,references)

    print(f'Evaluatioin results at Epoch {epoch}, BLEU-4: {bleu4}, Cider: {cider}, SPICE: {avg_spice_dict["spice"]}, ROUGE: {avg_rouge_dict["rouge"]}')
    return bleu4, cider



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    ''' ===for gridTD flickr30k ==='''
    # parser = imgcap_gridTD_argument_parser()
    # args = parser.parse_args()
    # args.dataset = 'flickr30k'
    # args.lrp_tune = False
    # args.hidden_dim = 1024
    # args.epochs = 10
    # args.decoder_lr = 0.00001
    # args.resume = glob.glob('./output/gridTD_BU/vgg16/flickr30k/checkpoint_flickr30k_epoch8_cider_0.5212386581763736.pth*')[0]
    # args.save_path = './output/gridTD_BU_baselinefinetune/'
    # main(args)
    #
    #
    # args.lrp_tune = True
    # args.hidden_dim = 1024
    # args.epochs = 10
    # args.decoder_lr = 0.00001
    # args.resume = glob.glob('./output/gridTD_BU/vgg16/flickr30k/checkpoint_flickr30k_epoch8_cider_0.5212386581763736.pth*')[0]
    # args.save_path = './output/gridTD_BU_lrpfinetune/'
    # main(args)

    '''===for gridTD coco2017==='''
    # parser = imgcap_gridTD_argument_parser()
    # args = parser.parse_args()
    # args.dataset = 'coco2017'
    # args.cider_tune = False
    # args.lrp_cider_tune = True
    # args.hidden_dim = 1024
    # args.epochs = 15
    # args.decoder_lr = 0.00005
    # args.save_path = './output/gridTD_BU_lrpcideropt'
    # args.resume = glob.glob('./output/gridTD_BU_cideropt/vgg16/coco2017/checkpoint_coco2017_epoch13_cider_1.1183009583739139.pth')[0]
    # main(args)
    ''' ===for aoa flickr30k ==='''
    parser = imgcap_aoa_argument_parser()
    args = parser.parse_args()
    args.dataset = 'flickr30k'
    # args.lrp_tune = False
    # args.hidden_dim = 1024
    # args.epochs = 20
    # args.decoder_lr = 0.00001
    # args.resume = glob.glob('./output/aoa_bu/vgg16/flickr30k/checkpoint_flickr30k_epoch12*')[0]
    # args.save_path = './output/aoa_bu/'
    # main(args)
    # #
    args.lrp_tune = True
    args.hidden_dim = 1024
    args.epochs = 15
    args.decoder_lr = 0.00001
    args.resume = glob.glob('./output/aoa_bu/vgg16/flickr30k/checkpoint_flickr30k_epoch13*')[0]
    args.save_path = './output/aoa_bu_lrpfinetune/'
    main(args)


    '''===for aoa coco2017 bu'''
    # parser = imgcap_aoa_argument_parser()
    # args = parser.parse_args()
    # args.dataset = 'coco2017'
    # args.lrp_tune = False
    # args.hidden_dim = 1024
    # args.epochs = 12
    # args.decoder_lr = 0.00001
    # args.resume = glob.glob('./output/aoa_bu/vgg16/coco2017/checkpoint_coco2017_epoch10_cider_1.0585677209425806.pth*')[0]
    # args.save_path = './output/aoa_bu_baselinefinetune'
    # main(args)
    #
    # args.lrp_tune = True
    # args.hidden_dim = 1024
    # args.epochs = 12
    # args.decoder_lr = 0.00001
    # args.resume = glob.glob('./output/aoa_bu/vgg16/coco2017/checkpoint_coco2017_epoch10_cider_1.0585677209425806.pth*')[0]
    # args.save_path = './output/aoa_bu_lrpfinetune'
    # main(args)


    '''===for gridTD cocorobust====='''
    # parser = imgcap_gridTD_argument_parser()
    # args = parser.parse_args()
    # args.dataset = 'coco2017'
    # args.lrp_tune = False
    # args.hidden_dim = 1024
    # args.epochs = 6
    # args.decoder_lr = 0.00001
    #
    # args.resume = glob.glob('./output/gridTD_BU/vgg16/coco2017/checkpoint_coco2017_epoch5_cider_1.0444984969172542.pth')[0]
    # args.save_path = './output/gridTD_BU_baselinefinetune/'
    # main(args)
    #
    # args.lrp_tune = False
    # args.hidden_dim = 1024
    # args.epochs = 6
    # args.decoder_lr = 0.00001
    # args.resume = glob.glob('./output/gridTD_BU/vgg16/coco2017/checkpoint_coco2017_epoch5_cider_1.0444984969172542.pth')[0]
    # args.save_path = './output/gridTD_BU_lrpfinetune/'
    # main(args)

    '''===for aoa cocorobust==='''
    # parser = imgcap_aoa_argument_parser()
    # args = parser.parse_args()
    # args.lrp_tune = False
    # args.cider_tune = False
    # args.lrp_cider_tune = False
    # args.finetune_encoder = False
    # args.hidden_dim = 1024
    # args.epochs = 15
    # args.decoder_lr = 0.0005
    # args.dataset = 'cocorobust'
    # args.save_path ='./output/aoa_bu/'
    # # args.resume = glob.glob('./output/gridTD_BU_cideropt/vgg16/cocorobust/checkpoint_cocorobust_epoch23*')[0]
    # main(args)

    '''for aoa coco'''
    # parser = imgcap_aoa_argument_parser()
    # args = parser.parse_args()
    # args.dataset = 'coco2017'
    # args.resume = glob.glob('./output/aoa/vgg16/coco2017/BEST_checkpoint_coco2017_epoch24*')[0]


    # main(args)