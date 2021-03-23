import os
import io
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy as np
import cv2
import torch
import PIL.Image
import json
import h5py
NUM_OBJECTS = 36

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)

def create_bu_features(dataset):
    if dataset == 'coco2017':
        train_data_path = './dataset/train_imagecap_coco2017_5_cap_per_img_4_min_word_freq.json'
        test_data_path = './dataset/test_imagecap_coco2017_5_cap_per_img_4_min_word_freq.json'
        val_data_path = './dataset/val_imagecap_coco2017_5_cap_per_img_4_min_word_freq.json'
    elif dataset == 'flickr30k':
        train_data_path = './dataset/train_imagecap_flickr30k_5_cap_per_img_3_min_word_freq.json'
        test_data_path = './dataset/test_imagecap_flickr30k_5_cap_per_img_3_min_word_freq.json'
        val_data_path = './dataset/val_imagecap_flickr30k_5_cap_per_img_3_min_word_freq.json'
    elif dataset == 'coco2014':
        train_data_path = './dataset/train_imagecap_coco2014_5_cap_per_img_4_min_word_freq.json'
        test_data_path = './dataset/test_imagecap_coco2014_5_cap_per_img_4_min_word_freq.json'
        val_data_path = './dataset/val_imagecap_coco2014_5_cap_per_img_4_min_word_freq.json'
    else:
        raise NotImplementedError('dataset in coco2017 or flickr30k')
    train_data = json.load(open(train_data_path,'r'))
    test_data = json.load(open(test_data_path, 'r'))
    val_data = json.load(open(val_data_path, 'r'))
    data_path = './demo/data/genome/1600-400-20'
    vg_classes = []
    with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
        for object in f.readlines():
            vg_classes.append(object.split(',')[0].lower().strip())

    vg_attrs = []
    with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
        for object in f.readlines():
            vg_attrs.append(object.split(',')[0].lower().strip())

    MetadataCatalog.get("vg").thing_classes = vg_classes
    MetadataCatalog.get("vg").attr_classes = vg_attrs

    cfg = get_cfg()
    cfg.merge_from_file("./configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    # VG Weight
    cfg.MODEL.WEIGHTS = "./demo/faster_rcnn_from_caffe_attr.pkl"
    predictor = DefaultPredictor(cfg)
    extract_featrue(predictor, 'train', dataset, train_data)
    extract_featrue(predictor, 'test', dataset, test_data)
    extract_featrue(predictor, 'val', dataset, val_data)

def extract_featrue(predictor, split, dataset_name, data, th=0.2):
    save_path = f'/home/sunjiamei/work/ImageCaptioning/dataset/{dataset_name}_bu_features/{split}'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    h5_file_name = []
    print(len(data))
    for idx,item in enumerate(data):
        print(idx, len(data))
        img_path = item['image_path']
        raw_image = cv2.imread(img_path)
        # im_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        image_filename = img_path.split('/')[-1]
        if image_filename not in h5_file_name:
            h5_file_name.append(image_filename)
        else:
            continue
        with torch.no_grad():
            raw_height, raw_width = raw_image.shape[:2]
            # print("Original image size: ", (raw_height, raw_width))
            # Preprocessing
            image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
            # print("Transformed image size: ", image.shape[:2])
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": raw_height, "width": raw_width}]
            images = predictor.model.preprocess_image(inputs)

            # Run Backbone Res1-Res4
            features = predictor.model.backbone(images.tensor)

            # Generate proposals with RPN
            proposals, _ = predictor.model.proposal_generator(images, features, None)
            proposal = proposals[0]
            # print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape)

            # Run RoI head for each proposal (RoI Pooling + Res5)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            features = [features[f] for f in predictor.model.roi_heads.in_features]
            box_features = predictor.model.roi_heads._shared_roi_transform(
                features, proposal_boxes
            )
            feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
            # print('Pooled features size:', feature_pooled.shape)

            # Predict classes and boxes for each proposal.
            pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(
                feature_pooled)
            outputs = FastRCNNOutputs(
                predictor.model.roi_heads.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                predictor.model.roi_heads.smooth_l1_beta,
            )
            probs = outputs.predict_probs()[0]
            boxes = outputs.predict_boxes()[0]

            attr_prob = pred_attr_logits[..., :-1].softmax(-1)
            max_attr_prob, max_attr_label = attr_prob.max(-1)

            # Note: BUTD uses raw RoI predictions,
            #       we use the predicted boxes instead.
            # boxes = proposal_boxes[0].tensor

            # NMS
            for nms_thresh in np.arange(0.5, 1.0, 0.1):
                instances, ids = fast_rcnn_inference_single_image(
                    boxes, probs, image.shape[1:],
                    score_thresh=th, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
                )
                if len(ids) == NUM_OBJECTS:
                    break
            instances = detector_postprocess(instances, raw_height, raw_width)
            roi_features = feature_pooled[ids].detach()
            max_attr_prob = max_attr_prob[ids].detach()
            max_attr_label = max_attr_label[ids].detach()
            instances.attr_scores = max_attr_prob
            instances.attr_classes = max_attr_label
            if roi_features.size(0)< NUM_OBJECTS:
                extract_single(img_path, save_path, image_filename)
                continue
            h5_save_file = h5py.File(os.path.join(save_path, image_filename+'.hdf5'), 'w')
            h5_save_file.create_dataset('image_features', data=roi_features.cpu().numpy())
            h5_save_file.create_dataset('image_boxes', data=instances.pred_boxes.tensor.cpu().numpy())
            h5_save_file.close()

            # read_file = h5py.File(os.path.join(save_path, image_filename+'.hdf5'), 'r')
            # img_features = read_file['image_features'][:]
            # img_boxes = read_file['image_boxes'][:]
            # print(img_features)
            # print(img_boxes)
        # break


def extract_single(raw_img_path, save_path, image_filename):
    data_path = './demo/data/genome/1600-400-20'
    vg_classes = []
    with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
        for object in f.readlines():
            vg_classes.append(object.split(',')[0].lower().strip())

    vg_attrs = []
    with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
        for object in f.readlines():
            vg_attrs.append(object.split(',')[0].lower().strip())

    MetadataCatalog.get("vg").thing_classes = vg_classes
    MetadataCatalog.get("vg").attr_classes = vg_attrs

    cfg = get_cfg()
    cfg.merge_from_file("./configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    # VG Weight
    cfg.MODEL.WEIGHTS = "./demo/faster_rcnn_from_caffe_attr.pkl"
    predictor = DefaultPredictor(cfg)

    raw_image = cv2.imread(raw_img_path)
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        # print("Original image size: ", (raw_height, raw_width))
        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        # print("Transformed image size: ", image.shape[:2])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)

        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]
        # print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        # print('Pooled features size:', feature_pooled.shape)

        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(
            feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]
        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)

        # Note: BUTD uses raw RoI predictions,
        #       we use the predicted boxes instead.
        # boxes = proposal_boxes[0].tensor

        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:],
                score_thresh=0.1, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
            )
            if len(ids) == NUM_OBJECTS:
                break
        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids].detach()
        max_attr_prob = max_attr_prob[ids].detach()
        max_attr_label = max_attr_label[ids].detach()
        instances.attr_scores = max_attr_prob
        instances.attr_classes = max_attr_label
        print(roi_features.size(), image_filename)
        if roi_features.size(0) == 0:
            print(image_filename)
            return
        print('roi_features and box size less', roi_features.size(), instances.pred_boxes.tensor.size())
        h5_save_file = h5py.File(os.path.join(save_path, image_filename + '.hdf5'), 'w')
        h5_save_file.create_dataset('image_features', data=roi_features.cpu().numpy())
        h5_save_file.create_dataset('image_boxes', data=instances.pred_boxes.tensor.cpu().numpy())
        h5_save_file.close()

if __name__ == '__main__':
    create_bu_features('coco2014')
    # 67001  28987 7973 101239
    # extract_single('000000334477.jpg')