import os
import json
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import nltk
from collections import defaultdict
from shutil import copyfile
class COCOCategory(object):
    def __init__(self, file_path):
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                dataset = json.load(f)      # ['images', 'licenses', 'annotations', 'info', 'categories']
                f.close()
            print(list(dataset.keys()))
            self.images = dataset['images']      # dict_keys(['license', 'coco_url', 'width', 'flickr_url', 'file_name', 'height', 'date_captured', 'id'])
            self.annotations = dataset['annotations']     # dict_keys(['iscrowd', 'area', 'image_id', 'segmentation', 'bbox', 'category_id', 'id'])
            self.categories = dataset['categories']  #['name', 'id', 'supercategory']
            print(self.categories)
            print(len(self.images))
        else:
            raise NotImplementedError('The input path is invalid')
        self.filename_to_category = dict()
        self._build_category()

    def _build_category(self):
        '''dict:  key - img_name, '''
        id_to_file_name = dict()
        for img_item in self.images:
            if img_item['id'] not in id_to_file_name.keys():
                id_to_file_name[img_item['id']] = dict()
            id_to_file_name[img_item['id']]['file_name'] = img_item['file_name']
            id_to_file_name[img_item['id']]['shape'] = (img_item['width'], img_item['height'])
        print(len(id_to_file_name))
        categoryid_to_categories = dict()
        for category_item in self.categories:
            categoryid_to_categories[category_item['id']] = category_item['name']
        imgid_to_categorynames = dict()
        imgid_to_bbbox = dict()
        for annotation_item in self.annotations:
            imgid = annotation_item['image_id']
            cate_id = annotation_item['category_id']
            if imgid not in imgid_to_categorynames.keys():
                imgid_to_categorynames[annotation_item['image_id']] = dict()
            imgid_to_categorynames[annotation_item['image_id']][categoryid_to_categories[cate_id]]=str(cate_id)
            if imgid not in imgid_to_bbbox.keys():
                imgid_to_bbbox[imgid] = dict()
            if cate_id not in imgid_to_bbbox[imgid].keys():
                imgid_to_bbbox[imgid][cate_id] = []
            coordinates= annotation_item['bbox']
            xmin = coordinates[0]
            ymin = coordinates[1]
            xmax = coordinates[2] + xmin
            ymax = coordinates[3] + ymin
            new_bbox = [xmin, ymin, xmax, ymax]
            imgid_to_bbbox[imgid][cate_id].append(new_bbox)  # bbox is [x, y, xmax, ymax] x is for width dimension
        print(len(imgid_to_bbbox))
        print(len(imgid_to_categorynames))
        for imgid in imgid_to_categorynames.keys():
            filename = id_to_file_name[imgid]['file_name']
            shape = id_to_file_name[imgid]['shape']
            ratio = (224/shape[0], 224/shape[1])
            self.filename_to_category[filename] = dict()
            self.filename_to_category[filename]['categories'] = imgid_to_categorynames[imgid]
            self.filename_to_category[filename]['bbox'] = imgid_to_bbbox[imgid]
            self.filename_to_category[filename]['shape'] = shape
            self.filename_to_category[filename]['resize_ratio'] = ratio
        print(len(self.filename_to_category))
        with open('./dataset/coco/COCOvalEntities.json', 'w') as f:
            json.dump(self.filename_to_category, f)
            f.close()
        with open('./dataset/coco/COCOvalEntities.json', 'r') as f:
            filedict = json.load(f)
            f.close()
        print(filedict.keys())
        for key in filedict.keys():
            for k in filedict[key].keys():
                print(k)
                print(filedict[key][k])
            break
        print(len(filedict))
        return self.filename_to_category


def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder='./',
                       max_len=100):
    """
    Creates input files for training, validation, and test data.
    Introduction of datafiles:
    the '.json' files have two keys 'images' and 'dataset'
    'dataset' contains a str indicating the dataset name
    'images' contains items with several keys: ['filepath', 'sentids', 'filename', 'imgid', 'split', 'sentences', 'cocoid'],
    under the key 'sentences' is a list of sentences with ['tokens'(a list of words), 'raw'(a sentence string), 'imgid', 'sentid']
    For flickr datasets, there is no cocoid
    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco2014', 'flickr8k', 'flickr30k', 'coco2017'}

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
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            if img['split'] in ['train', 'restval']:
                word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if 'coco' in dataset else os.path.join(
            image_folder, img['filename'])
        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] >= min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0
    print('vocab_size is ', len(word_map))
    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'wordmap_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'train'),
                                   (val_image_paths, val_image_captions, 'val'),
                                   (test_image_paths, test_image_captions, 'test')]:
        imgcapdata = []
        for i, path in enumerate(tqdm(impaths)):
            # print(path)
            assert os.path.isfile(path)
            enc_captions = []
            caplens = []
            # Sample captions
            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
            else:
                captions = sample(imcaps[i], k=captions_per_image)
            # Sanity check
            assert len(captions) == captions_per_image
            for j, c in enumerate(captions):
                # Encode captions
                enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                    word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                # Find caption lengths
                c_len = len(c) + 2
                enc_captions.append(enc_c)
                caplens.append(c_len)
                # print(path, enc_c, c_len)
            assert len(enc_captions) == len(caplens)
            assert len(enc_captions) == len(captions)
            if split == 'train':
                for idx in range(captions_per_image):
                    item = {'image_path':path, 'encoded_cap':enc_captions[idx],'encoded_all_caps':enc_captions, 'caption_len': caplens[idx]}
                    imgcapdata.append(item)
            else:
                item = {'image_path': path, 'encoded_all_caps': enc_captions,'caption_len': caplens}
                imgcapdata.append(item)
        print(f'{split} length is {len(imgcapdata)}')
        with open(os.path.join(output_folder, split + '_imagecap_' + base_filename + '.json'), 'a') as h:
            json.dump(imgcapdata, h)


def create_input_robust_coco(karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder='./',
                       max_len=100):
    """
    Creates input files for training, validation, and test data.
    Introduction of datafiles:
    the '.json' files have two keys 'images' and 'dataset'
    'dataset' contains a str indicating the dataset name
    'images' contains items with several keys: ['filepath', 'sentids', 'filename', 'imgid', 'split', 'sentences', 'cocoid'],
    under the key 'sentences' is a list of sentences with ['tokens'(a list of words), 'raw'(a sentence string), 'imgid', 'sentid']
    For flickr datasets, there is no cocoid
    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """
    def clean_id(data_list):
        id_list = []
        for item in data_list:
            id_list.append(item['img_id'])
        assert len(id_list) == len(data_list)
        return list(set(id_list))
    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)
    with open('split_robust_coco.json', 'r') as f:
        robust_split = json.load(f)
    train_ids = clean_id(robust_split['train_id'])
    val_ids = clean_id(robust_split['val_id'])
    test_ids = clean_id(robust_split['test_id'])
    print(len(train_ids), len(val_ids), len(test_ids))
    print(len(data['images']), data['images'][0].keys())
    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        if img['cocoid'] in train_ids:
            split_flag = 'train'
        elif img['cocoid'] in test_ids:
            split_flag = 'test'
        elif img['cocoid'] in val_ids:
            split_flag = 'val'
        else:
            continue
        for c in img['sentences']:
            # Update word frequency
            if split_flag in ['train', 'restval']:
                word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename'])
        if split_flag in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif split_flag in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif split_flag in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] >= min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0
    print('vocab_size is ', len(word_map))
    # Create a base/root name for all output files
    base_filename = 'cocorobust' + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'wordmap_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'train'),
                                   (val_image_paths, val_image_captions, 'val'),
                                   (test_image_paths, test_image_captions, 'test')]:
        imgcapdata = []
        for i, path in enumerate(tqdm(impaths)):
            # print(path)
            assert os.path.isfile(path)
            enc_captions = []
            caplens = []
            # Sample captions
            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
            else:
                captions = sample(imcaps[i], k=captions_per_image)
            # Sanity check
            assert len(captions) == captions_per_image
            for j, c in enumerate(captions):
                # Encode captions
                enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                    word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                # Find caption lengths
                c_len = len(c) + 2
                enc_captions.append(enc_c)
                caplens.append(c_len)
                # print(path, enc_c, c_len)
            assert len(enc_captions) == len(caplens)
            assert len(enc_captions) == len(captions)
            if split == 'train':
                for idx in range(captions_per_image):
                    item = {'image_path':path, 'encoded_cap':enc_captions[idx],'encoded_all_caps':enc_captions, 'caption_len': caplens[idx]}
                    imgcapdata.append(item)
            else:
                item = {'image_path': path, 'encoded_all_caps': enc_captions,'caption_len': caplens}
                imgcapdata.append(item)
        print(f'{split} length is {len(imgcapdata)}')
        with open(os.path.join(output_folder, split + '_imagecap_' + base_filename + '.json'), 'w') as h:
            json.dump(imgcapdata, h)

def create_input_files_noc(dataset, karpathy_json_path, held_out_lists_folder,image_folder, captions_per_image, min_word_freq, output_folder='./',
                       max_len=100):
    """
    Creates input files for training, validation, and test data.
    Introduction of datafiles:
    the '.json' files have two keys 'images' and 'dataset'
    'dataset' contains a str indicating the dataset name
    'images' contains items with several keys: ['filepath', 'sentids', 'filename', 'imgid', 'split', 'sentences', 'cocoid'],
    under the key 'sentences' is a list of sentences with ['tokens'(a list of words), 'raw'(a sentence string), 'imgid', 'sentid']
    For flickr datasets, there is no cocoid
    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param held_out_lists_folder: the .txt file path for held_out train/val/test/ split
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco2014_held_out'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)
    # Read held-out-split-ids
    train_ids = []
    test_ids = []
    val_ids = []
    with open(os.path.join(held_out_lists_folder, 'coco2014_cocoid.train.txt')) as f:
        train_file = f.readlines()
        for line in train_file:
            train_ids.append(int(line.strip('\n')))
    with open(os.path.join(held_out_lists_folder, 'coco2014_cocoid.val_val.txt')) as f:
        val_file = f.readlines()
        for line in val_file:
            val_ids.append(int(line.strip('\n')))
    with open(os.path.join(held_out_lists_folder, 'coco2014_cocoid.val_test.txt')) as f:
        test_file = f.readlines()
        for line in test_file:
            test_ids.append(int(line.strip('\n')))
    print(len(train_ids), len(test_ids), len(val_ids))
    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if 'coco' in dataset else os.path.join(
            image_folder, img['filename'])
        # print(img['cocoid'])
        if int(img['cocoid']) in train_ids:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif int(img['cocoid']) in val_ids:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif int(img['cocoid']) in test_ids:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] >= min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0
    print('vocab_size is ', len(word_map))
    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'wordmap_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'train'),
                                   (val_image_paths, val_image_captions, 'val'),
                                   (test_image_paths, test_image_captions, 'test')]:
        imgcapdata = []
        for i, path in enumerate(tqdm(impaths)):
            # print(path)
            assert os.path.isfile(path)
            enc_captions = []
            caplens = []
            # Sample captions
            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
            else:
                captions = sample(imcaps[i], k=captions_per_image)
            # Sanity check
            assert len(captions) == captions_per_image
            for j, c in enumerate(captions):
                # Encode captions
                enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                    word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                # Find caption lengths
                c_len = len(c) + 2
                enc_captions.append(enc_c)
                caplens.append(c_len)
                # print(path, enc_c, c_len)
            assert len(enc_captions) == len(caplens)
            assert len(enc_captions) == len(captions)
            if split == 'train':
                for idx in range(captions_per_image):
                    item = {'image_path':path, 'encoded_cap':enc_captions[idx],'encoded_all_caps':enc_captions, 'caption_len': caplens[idx]}
                    imgcapdata.append(item)
            else:
                item = {'image_path': path, 'encoded_all_caps': enc_captions,'caption_len': caplens}
                imgcapdata.append(item)
        print(f'{split} length is {len(imgcapdata)}')
        with open(os.path.join(output_folder, split + '_imagecap_' + base_filename + '.json'), 'a') as h:
            json.dump(imgcapdata, h)


def generate_coco2017_jsonfile():
    dataset = {}
    dataset['dataset'] = 'coco2017'
    dataset['images'] = []
    annotation_train_file_path = '/home/sunjiamei/work/ImageCaptioning/dataset/coco/annotations/captions_train2017.json'
    annotation_val_file_path = '/home/sunjiamei/work/ImageCaptioning/dataset/coco/annotations/captions_val2017.json'
    coco_anns_train = json.load(open(annotation_train_file_path, 'r')) # keys are info, licenses, images, annotations  118287
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    # new_words = tokenizer.tokenize(sentence)
    # under annotations, keys are  image_id, id(the sentence id), caption
    # under images, the keys are license, file_name, coco_url, height, width, data_captured, flickr_url id
    imgID2captions_train = defaultdict(dict)
    imgID2imgFilename_path_train = defaultdict(dict)
    for ann in coco_anns_train['annotations']:
        caption_str = ann['caption'].lower()
        tokens = tokenizer.tokenize(caption_str)
        # print(tokens)
        if not 'sentids' in imgID2captions_train[ann['image_id']].keys():
            imgID2captions_train[ann['image_id']]['sentids'] = []
            imgID2captions_train[ann['image_id']]['sentences'] = []
        imgID2captions_train[ann['image_id']]['sentids'].append(ann['id'])
        imgID2captions_train[ann['image_id']]['sentences'].append({'tokens':tokens, 'raw': caption_str, 'imgid': ann['image_id'],
                                                             'sentid': ann['id']})
    for img in coco_anns_train['images']:
        imgID2imgFilename_path_train[img['id']]['filename'] = img['file_name']
        imgID2imgFilename_path_train[img['id']]['filepath'] = 'train2017'
    imgID2imgFilename_path_train = dict(imgID2imgFilename_path_train)
    imgID2captions_train = dict(imgID2captions_train)
    img_ids_trainval = list(imgID2captions_train.keys())
    img_ids_trainval.sort()

    coco_anns_val = json.load(open(annotation_val_file_path, 'r'))
    imgID2captions_val = defaultdict(dict)
    imgID2imgFilename_path_val = defaultdict(dict)
    for ann in coco_anns_val['annotations']:
        caption_str = ann['caption'].lower()
        tokens = tokenizer.tokenize(caption_str)
        # print(tokens)
        if not 'sentids' in imgID2captions_val[ann['image_id']].keys():
            imgID2captions_val[ann['image_id']]['sentids'] = []
            imgID2captions_val[ann['image_id']]['sentences'] = []
        imgID2captions_val[ann['image_id']]['sentids'].append(ann['id'])
        imgID2captions_val[ann['image_id']]['sentences'].append({'tokens':tokens, 'raw': caption_str, 'imgid': ann['image_id'],
                                                             'sentid': ann['id']})
    for img in coco_anns_val['images']:
        imgID2imgFilename_path_val[img['id']]['filename'] = img['file_name']
        imgID2imgFilename_path_val[img['id']]['filepath'] = 'val2017'
    imgID2imgFilename_path_val = dict(imgID2imgFilename_path_val)
    imgID2captions_val = dict(imgID2captions_val)
    img_ids_test= list(imgID2captions_val.keys())


    for i in range(len(img_ids_trainval)):
        if i <110000:
            split = 'train'
        else:
            split = 'val'
        img_id = img_ids_trainval[i]
        dataunit = {}
        dataunit["filepath"] = imgID2imgFilename_path_train[img_id]['filepath']
        dataunit["filename"] = imgID2imgFilename_path_train[img_id]['filename']
        dataunit["sentids"] = imgID2captions_train[img_id]['sentids']
        dataunit["sentences"] = imgID2captions_train[img_id]['sentences']
        dataunit["imgid"] = img_id
        dataunit["cocoid"] = img_id
        dataunit["split"] = split
        dataset['images'].append(dataunit)

    for i in range(len(img_ids_test)):
        split = 'test'
        img_id = img_ids_test[i]
        dataunit = {}
        dataunit["filepath"] = imgID2imgFilename_path_val[img_id]['filepath']
        dataunit["filename"] = imgID2imgFilename_path_val[img_id]['filename']
        dataunit["sentids"] = imgID2captions_val[img_id]['sentids']
        dataunit["sentences"] = imgID2captions_val[img_id]['sentences']
        dataunit["imgid"] = img_id
        dataunit["cocoid"] = img_id
        dataunit["split"] = split
        dataset['images'].append(dataunit)
    with open('./dataset_coco2017.json', 'w') as f:
        json.dump(dataset, f)


def generate_coco2017_data():
    '''
        vocab_size:
        train_data_length:
        val_data_length: 5000
        test_data_length: 5000
    :return:
    '''
    dataset = 'coco2017'
    json_path = 'dataset_coco2017.json'
    image_folder = '/home/sunjiamei/work/ImageCaptioning/dataset/coco/images'
    caption_per_img = 5
    min_freq_word = 4
    create_input_files(dataset, karpathy_json_path=json_path, image_folder=image_folder,
                       captions_per_image=caption_per_img, min_word_freq=min_freq_word, max_len=30)


def generate_flickr30k_data():
    '''
    vocab_size: 9586
    train_data_length: 29000  number of img_caption pair for training:145000
    val_data_length: 1014
    test_data_length: 1000
    :return: a json file with a list of [{'image_path':path, 'encode_caps':enc_captions, 'caption_len': caplens}]
    '''
    dataset = 'flickr30k'
    json_path = 'dataset_flickr30k.json'
    image_folder = '/home/sunjiamei/work/ImageCaptioning/dataset/flickr30k/Flickr30k_Dataset'
    caption_per_img = 5
    min_freq_word = 3
    create_input_files(dataset, karpathy_json_path=json_path, image_folder=image_folder,
                       captions_per_image=caption_per_img, min_word_freq=min_freq_word, max_len=50)


def generate_coco2014_data():
    '''
        vocab_size: 11143
        train_data_length: 113287  number of img_caption pair for training:566435
        val_data_length: 5000
        test_data_length: 5000
    :return:
    '''
    dataset = 'coco2014'
    json_path = 'dataset_coco.json'
    image_folder = '/home/sunjiamei/work/ImageCaptioning/dataset/coco/images'
    caption_per_img = 5
    min_freq_word = 4
    create_input_files(dataset, karpathy_json_path=json_path, image_folder=image_folder,
                       captions_per_image=caption_per_img, min_word_freq=min_freq_word, max_len=50)


def generate_coco2014_held_out_data():
    '''
        vocab_size: 11569
        train_data_length: 82783  number of img_caption pair for training:413915
        val_data_length: 20252
        test_data_length: 20252
    :return:
    '''
    dataset = 'coco2014_held_out'
    json_path = 'dataset_coco.json'
    image_folder = '/home/sunjiamei/work/ImageCaptioning/dataset/coco/images'
    held_out_lists_folder = './image_list'
    caption_per_img = 5
    min_freq_word = 4
    create_input_files_noc(dataset, karpathy_json_path=json_path, held_out_lists_folder=held_out_lists_folder, image_folder=image_folder,
                       captions_per_image=caption_per_img, min_word_freq=min_freq_word, max_len=35)


def generate_robust_coco():
    json_path = 'dataset_coco.json'
    image_folder = '/home/sunjiamei/work/ImageCaptioning/dataset/coco/images'
    caption_per_img = 5
    min_freq_word = 4
    create_input_robust_coco(karpathy_json_path=json_path, image_folder=image_folder, captions_per_image=caption_per_img, min_word_freq=min_freq_word, max_len=50)


def copy_bu_features_robust_coco():

    def clean_id(data_list):
        id_list = []
        for item in data_list:
            id_list.append(item['img_id'])
        assert len(id_list) == len(data_list)
        return list(set(id_list))
    root_dir = '/home/sunjiamei/work/ImageCaptioning/dataset/cocorobust_bu_feature'
    coco_2014_bu_dir = '/home/sunjiamei/work/ImageCaptioning/dataset/coco2014_bu_features'
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')
    val_dir = os.path.join(root_dir, 'val')
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
        os.makedirs(train_dir)
        os.makedirs(test_dir)
        os.makedirs(val_dir)
    with open('dataset_coco.json', 'r') as j:
        data = json.load(j)
    with open('split_robust_coco.json', 'r') as f:
        robust_split = json.load(f)
    train_ids = clean_id(robust_split['train_id'])
    val_ids = clean_id(robust_split['val_id'])
    test_ids = clean_id(robust_split['test_id'])
    print(len(train_ids), len(val_ids), len(test_ids))
    for img in data['images']:
        captions = []
        if img['cocoid'] in train_ids:
            split_flag = 'train'
        elif img['cocoid'] in test_ids:
            split_flag = 'test'
        elif img['cocoid'] in val_ids:
            split_flag = 'val'
        else:
            continue
        # if img['split'] in ['train', 'restval']:
        #     copyfile(os.path.join(coco_2014_bu_dir,'train',img['filename']+'.hdf5'), os.path.join(root_dir, split_flag, img['filename']+'.hdf5'))
        if img['split'] in ['val']:
            copyfile(os.path.join(coco_2014_bu_dir, 'val', img['filename'] + '.hdf5'),
                     os.path.join(root_dir, split_flag, img['filename'] + '.hdf5'))
        elif img['split'] in ['test']:
            copyfile(os.path.join(coco_2014_bu_dir, 'test', img['filename'] + '.hdf5'),
                     os.path.join(root_dir, split_flag, img['filename'] + '.hdf5'))
        else:
            continue



# if __name__ == '__main__':
    # generate_flickr30k_data()
    # generate_coco2014_data()
    # generate_coco2017_jsonfile()
    # generate_coco2017_data()
    # generate_coco2014_held_out_data()
    # generate_robust_coco()
    # copy_bu_features_robust_coco()
