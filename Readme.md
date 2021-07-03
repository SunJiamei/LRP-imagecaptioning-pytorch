# LRP ImageCaptioning Pytorch
This is a Pytorch implementation of the latest version of [Understanding Image Captioning Model beyond Visualizing Attention](https://arxiv.org/abs/2001.01037)

![](./examples/imgs/graphabstract.jpg)
![](./examples/imgs/sport.png)
![](./examples/imgs/sport_sentence.png)


### What can we do with this repo
1. To train image captioning models with two kinds of attention mechanisms, adaptive attention, and multi-head attention.
2. To get both image explanations and linguistic explanations for a predicted word using LRP, Grad-CAM, Guided Grad-CAM, and GuidedBackpropagation.
3. To fine-tune a pre-trained image captioning model with *LRP-inference fine-tuning* to improve the mAP of frequent object words.



### Requirements
python >=3.6 
pytorch =1.4.0

### Dataset Preparation
##### Flickr30K
We prepare the Flick30K as the Karpathy split. 
##### MSCOCO2017
We select 110000 images from the training set for training and 5000 images from the training set for validation. The original validation set is used for testing.

The vocabulary is built on the training set for both datasets. Each caption is encoded with a `<start>` token at the beginning and an `<end>` token at the end.
For the words that appear less than 3/4 time for Flicker30K and MSCOCO2017, we encode them with an `<unk>` token. 

To build the vocabulary and encode the reference captions, please refer to [preparedataset.py](./dataset/preparedataset.py).

### Feature extraction
This repo experiments with both the CNN features and the bottom-up features. 
The CNN features are extracted from the pre-trained VGG16 on ImageNet.
We follow the [py-bottom-up-attention](https://github.com/airsplay/py-bottom-up-attention.git) to extract 36 bottom-up features per image for training. 


### To Train Models From Scratch
We train the image captioning models with two attention mechanisms, the adaptive attention with an LSTM layer as the predictor
and multi-head attention with an `FC` layer as the predictor. 
The two models are defined in [gridTDmodel.py](./models/gridTDmodel.py) and [aoamodel.py](./models/aoamodel.py) respectively.


### Pre-trained Models
Our pre-trained models can be downloaded [here](https://drive.google.com/file/d/13PrwflX7mW48Lj7JN51cQ6-ZFBe83bEV/view?usp=sharing).
Please email to sunjiamei.hit@gmail.com if you could not access them.

  
### To Evaluate the Image Captioning Model
We evaluate the image captioning models using BLEU, SPICE, ROUGE, METEOR, and CIDER metrics. We also use [BERT score](https://pypi.org/project/bert-score/). To generate these evaluations,
we need to download the [pycocoevalcap](https://github.com/salaniz/pycocoevalcap.git) tools and copy the folders of different metrics under [./pycocoevalcap](pycocoevalcap). 
We already provide the `bert` folder. 

We provide three decoding methods:
1. greedy search
2. beam search
3. [diverse beam search](https://arxiv.org/abs/1610.02424) 
 

### To Explain Image Captioning Models
We provide LRP, GradCAM, Guided-GradCAM, and Guided Backpropagation to explain the image captioning models. 
These explanation methods are defined under the corresponding model files.

There are two stages of explanation. We first explain the decoder to get the explanation of each proceeding word and the encoded image features.
We then explain the image encoder to obtain the image explanations.


### To Fine-tune the Model with LRP Inference
We provide three optimization methods to optimize image captioning models trained with cross-entropy loss:
1. --cider_tune: the SCST optimization on a pre-trained model
2. --lrp_cider_tune: the lrp-inference SCST optimization 
3. --lrp_tune: the lrp-inference finetune with cross-entropy loss

### To Evaluate the Explanations

Please refer to the examples in [evaluatioin.py](evaluation.py). 
This will generate the results of our ablation experiment and *correctness* scores across various explanation methods.
we need to download the [COCOvalEntities.json](https://drive.google.com/file/d/1ygSGtJ79FyocW-QshgeuIEQlu24QgF0x/view?usp=sharing) file for calculating the *correctness* scores.
 



Acknowledgment
---------------
Many thanks to the works:

[a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning.git)

[AoANet](https://github.com/husthuaan/AoANet.git)

[py-bottom-up-attention](https://github.com/airsplay/py-bottom-up-attention.git)

[iNNvestigate](https://github.com/albermax/innvestigate.git)

[pycocoevalcap](https://github.com/salaniz/pycocoevalcap.git)



