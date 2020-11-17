from pycocoevalcap.bleu import bleu
from pycocoevalcap.cider import cider
from pycocoevalcap.meteor import meteor
from pycocoevalcap.rouge import rouge
from pycocoevalcap.spice import spice
from pycocoevalcap.bert import bert
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import os

class Score(object):
    """A subclass of this class is an adapter of pycocoevalcap."""

    def __init__(self, score_name, implementation):
        self._score_name = score_name
        self._implementation = implementation
        self.tokenizer = PTBTokenizer()

    def calculate(self, id_to_prediction, id_to_references):
        # id_to_preds = {}
        # for id_, pred in id_to_prediction.items():
        #     id_to_preds[id_] = [pred]
        id_to_references = self.tokenizer.tokenize(id_to_references)
        id_to_prediction = self.tokenizer.tokenize(id_to_prediction)
        avg_score, scores = self._implementation.compute_score(
                                                id_to_references, id_to_prediction)
        if isinstance(avg_score, (list, tuple)):
            avg_score = map(float, avg_score)
        else:
            avg_score = float(avg_score)
        return {self._score_name: avg_score}


class BLEU(Score):
    def __init__(self, n=4):
        implementation = bleu.Bleu(n)
        super(BLEU, self).__init__('bleu', implementation)
        self._n = n

    def calculate(self, id_to_prediction, id_to_references):

        name_to_score = super(BLEU, self).calculate(id_to_prediction,
                                                    id_to_references)
        scores = list(name_to_score.values())[0]
        result = {}
        for i, score in enumerate(scores, start=1):
            name = '{}_{}'.format(self._score_name, i)
            result[name] = score
        return result


class CIDEr(Score):
    def __init__(self):
        implementation = cider.Cider()
        super(CIDEr, self).__init__('cider', implementation)


class METEOR(Score):
    def __init__(self):
        implementation = meteor.Meteor()
        super(METEOR, self).__init__('meteor', implementation)

    def calculate(self, id_to_prediction, id_to_references):
        if self._data_downloaded():
            return super(METEOR, self).calculate(id_to_prediction,
                                                 id_to_references)
        else:
            return {self._score_name: 0.0}

    def _data_downloaded(self):
        meteor_dir = os.path.dirname(meteor.__file__)
        return (os.path.isfile(os.path.join(meteor_dir, 'meteor-1.5.jar')) and
                os.path.isfile(
                        os.path.join(meteor_dir, 'data', 'paraphrase-en.gz')))


class ROUGE(Score):
    def __init__(self):
        implementation = rouge.Rouge()
        super(ROUGE, self).__init__('rouge', implementation)

class SPICE(Score):
    def __init__(self):
        implementation = spice.Spice()
        super(SPICE, self).__init__('spice', implementation)

class BERT(Score):
    def __init__(self):
        implementation = bert.Bert()
        super(BERT,self).__init__('bert', implementation)

