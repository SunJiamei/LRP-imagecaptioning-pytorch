import bert_score

class Bert:
    """
    Main Class to compute the bert metric

    """
    def __init__(self, lang='en', return_hash=False):
        # set cider to sum over 1 to 4-grams
        self._lang = lang
        # set the standard deviation parameter for gaussian penalty
        self._retrun_hash = return_hash

    def compute_score(self, gts, res):
        """
        Main function to compute bert score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: bert (float) : computed bert score for the corpus
        """

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        gt_list = []
        res_list = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]
            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)
            for i in range(len(ref)):
                gt_list.append(ref[i])
                assert isinstance(ref[i],str)
                res_list.append(hypo[0])
                assert isinstance(hypo[0], str)
        assert(len(gt_list) == len(res_list))
        assert isinstance(gt_list, list)
        assert isinstance(res_list, list)
        P, R, scores = bert_score.score(res_list, gt_list, lang=self._lang, return_hash=self._retrun_hash)
        score = scores.mean()
        scores = scores.numpy()
        score = score.numpy()
        return score, scores

    def method(self):
        return "BERT"