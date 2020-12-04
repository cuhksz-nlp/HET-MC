from rouge import rouge_scorer

class Evaluation:
    def __init__(self):
        self.metrics = ['rouge1', 'rouge2', 'rougeL']
        self.main_metric = 'rougeL'
        self.scorer = rouge_scorer.RougeScorer(self.metrics, use_stemmer=True)

    def rouge_score(self, y_pred_list, y_list):
        rouge_list = []
        overall_rouge = {metric: {'p': 0, 'r': 0, 'f': 0} for metric in self.metrics}
        for y_pred, y in zip(y_pred_list, y_list):
            score = self.scorer.score(y_pred, y)
            for metric in self.metrics:
                overall_rouge[metric]['p'] += score[metric].precision
                overall_rouge[metric]['r'] += score[metric].recall
                overall_rouge[metric]['f'] += score[metric].fmeasure
            rouge_list.append(score)

        for metric in self.metrics:
            for key in overall_rouge[metric].keys():
                overall_rouge[metric][key] = overall_rouge[metric][key] / (len(y_pred_list) + 1e-10)
        overall_main_metric = overall_rouge[self.main_metric]['f']
        return overall_rouge, rouge_list, overall_main_metric

    def get_evaluate_list(self, gold_all, pred_all, target, use_subset=False):
        # use_subset: will use the subset that contains both SUM2A and SUM2B
        selected_gold_list = []
        selected_pred_list = []
        # if target == 'SUM1':
        #     for gold_item, pred_item in zip(gold_all, pred_all):
        #         if gold_item['SUM1_ORIG']:
        #             selected_gold_list.append(gold_item['SUM1_ORIG'])
        #             selected_pred_list.append(pred_item)
        #         elif gold_item['SUM1']:
        #             selected_gold_list.append(gold_item['SUM1'])
        #             selected_pred_list.append(pred_item)
        # elif target in ['SUM2_ORIG', 'SUM2A', 'SUM2B']:
        #     for gold_item, pred_item in zip(gold_all, pred_all):
        #         if gold_item[target]:
        #             selected_gold_list.append(gold_item[target])
        #             selected_pred_list.append(pred_item)
        # else:
        #     raise ValueError()
        if use_subset:
            for gold_item, pred_item in zip(gold_all, pred_all):
                if not gold_item['SUM2A'] == '' and not gold_item['SUM2B'] == '':
                    selected_gold_list.append(gold_item[target])
                    selected_pred_list.append(pred_item)
        else:
            for gold_item, pred_item in zip(gold_all, pred_all):
                if not gold_item[target] == '':
                    selected_gold_list.append(gold_item[target])
                    selected_pred_list.append(pred_item)
        return selected_gold_list, selected_pred_list

    def rouge_log(self, results_dict):
        log_str = ""
        for x in ["1", "2", "l"]:
            log_str += "\nROUGE-%s:\n" % x
            for y in ["f_score", "recall", "precision"]:
                key = "rouge_%s_%s" % (x, y)
                key_cb = key + "_cb"
                key_ce = key + "_ce"
                val = results_dict[key]
                val_cb = results_dict[key_cb]
                val_ce = results_dict[key_ce]
                log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
        return log_str
