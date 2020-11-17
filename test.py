import json
import torch
from config import imgcap_adaptive_argument_parser, imgcap_gridTD_argument_parser, imgcap_aoa_argument_parser
import torchvision.transforms as transforms
from dataset.dataloader import ImagecapDataset
from models import aoamodel
from models import adaptiveattention
from models import gridTDmodel
from models.metrics import BLEU, CIDEr, BERT, SPICE, ROUGE,METEOR
import os
import yaml
def main(beam_search_type, args):


    print(f'The arguments are')
    print(args)
    word_map_path = f'./dataset/wordmap_{args.dataset}.json'
    word_map = json.load(open(word_map_path, 'r'))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    val_transform = transforms.Compose([
        transforms.Resize(size=(args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    print('==========Loading Data==========')
    val_data = ImagecapDataset(args.dataset, args.test_split, val_transform, )
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.workers,pin_memory=True)
    print(len(val_loader))
    print('==========Data Loaded==========')
    print('==========Setting Model==========')
    if args.model_type == 'adaptive':
        model = adaptiveattention.AdaptiveAttentionCaptioningModel(args.embed_dim, args.hidden_dim, len(word_map), args.encoder)

    elif args.model_type == 'gridtd':
        model = gridTDmodel.GridTDModel(args.embed_dim, args.hidden_dim, len(word_map), args.encoder)
    elif args.model_type == 'aoa':
        model = aoamodel.AOAModel(args.embed_dim, args.hidden_dim, args.num_head, len(word_map), args.encoder)
    else:
        raise NotImplementedError(f'model_type {args.model_type} does not available yet')
    model.cuda()

    if args.weight:
        print(f'==========Resuming weights from {args.weight}==========')
        checkpoint = torch.load(args.weight)
        start_epoch = checkpoint['epoch']
        # epochs_since_improvement = checkpoint['epochs_since_improvement']
        # best_cider = checkpoint['cider']
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print(f'==========Initializing model from random==========')
        start_epoch = 0
        epochs_since_improvement = 0
        best_cider = 0
    print(f'==========Start Testing==========')
    validate(val_loader, model, word_map, args, beam_search_type=beam_search_type, start_epoch=start_epoch)

def validate(val_loader, model, word_map, args, start_epoch, beam_search_type='dbs', beam_size=3):
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
    # print(references)
    results_dict = {}
    print("Calculating Evalaution Metric Scores......\n")
    avg_bleu_dict = BLEU().calculate(hypotheses,references)
    bleu4 = avg_bleu_dict['bleu_4']
    avg_cider_dict = CIDEr().calculate(hypotheses, references)
    cider = avg_cider_dict['cider']
    avg_bert_dict = BERT().calculate(hypotheses,references)
    bert = avg_bert_dict['bert']
    avg_spice_dict = SPICE().calculate(hypotheses, references)
    avg_rouge_dict = ROUGE().calculate(hypotheses,references)
    avg_meteor_dict = METEOR().calculate(hypotheses, references)
    print(f'Evaluatioin results, BLEU-4: {bleu4}, Cider: {cider}, SPICE: {avg_spice_dict["spice"]}, ROUGE: {avg_rouge_dict["rouge"]}')
    results_dict.update(avg_bert_dict)
    results_dict.update(avg_bleu_dict)
    results_dict.update(avg_cider_dict)
    results_dict.update(avg_rouge_dict)
    results_dict.update(avg_spice_dict)
    results_dict.update(avg_meteor_dict)


    # write the predictions and ground truth to files
    prediction_filename = f'predictions_{args.dataset}_split_{args.test_split}_{beam_search_type}_{beam_size}_epoch{start_epoch}.yaml'
    gt_filename = f'reference_{args.dataset}_split_{args.test_split}_{beam_search_type}_{beam_size}.yaml'
    with open(os.path.join(args.save_path, args.encoder, args.dataset, 'lrpfinetune1',prediction_filename), 'w') as f:
        yaml.safe_dump(prediction_save, f)
        f.close()
    with open(os.path.join(args.save_path, args.encoder, args.dataset,'lrpfinetune1',gt_filename), 'w') as f:
        yaml.safe_dump(gt_save, f)
        f.close()
    metrics_filename =f'metrics_{args.dataset}_split_{args.test_split}_{beam_search_type}_{beam_size}_epoch{start_epoch}.yaml'
    # write the evaluate metrics to files
    with open(os.path.join(args.save_path, args.encoder, args.dataset, 'lrpfinetune1', metrics_filename), 'w') as f:
        yaml.safe_dump(results_dict, f)
        f.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    import glob
    # model_weight_path = './output/adaptive/BEST_checkpoint_flickr30k_epoch49_cider_0.5414486487400703.pth'
    # model_weight_paths = glob.glob('./output/gridTD/vgg16/coco2017/BEST_checkpoint_coco2017_epoch22*')
    # model_weight_paths = glob.glob('./output/gridTD/vgg16/flickr30k/baselineciderfinetune/checkpoint_flickr30k_epoch28_cider_0.4996554402847656_baseline.pth')
    # model_weight_paths = glob.glob('./output/aoa/vgg16/flickr30k/BEST_checkpoint_flickr30k_epoch31*')
    # model_weight_paths = glob.glob('./output/aoa/vgg16/coco2017/BEST_checkpoint_coco2017_epoch34*')
    model_weight_paths = glob.glob('./output/gridTD/vgg16/coco2017/lrpfinetune1/checkpoint*')
    print(model_weight_paths)
    parser = imgcap_gridTD_argument_parser()
    # parser = imgcap_aoa_argument_parser()
    args = parser.parse_args()

    args.dataset = 'coco2017'
    # args.dataset = 'flickr30k'
    for model_weight_path in model_weight_paths:
        args.weight = model_weight_path
        main(beam_search_type='beam_search', args=args)