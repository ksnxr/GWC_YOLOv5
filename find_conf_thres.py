import pandas as pd
import os
import argparse
from pathlib import Path
import copy
import time
import numpy as np
from tqdm import tqdm


def parse_box(line):
    _, x_center, y_center, width, height, conf = list(map(lambda x: float(x), line.split(' ')))
    x_center *= 1024
    y_center *= 1024
    width *= 1024
    height *= 1024

    def clip(value):
        if value < 0:
            return 0
        elif value > 1024:
            return 1024
        else:
            return value

    x_min = clip(int(x_center - width / 2))
    y_min = clip(int(y_center - height / 2))
    x_max = clip(int(x_center + width / 2))
    y_max = clip(int(y_center + height / 2))

    return [x_min, y_min, x_max, y_max], float(conf)


def iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    if xA < xB and yA < yB:
        interArea = (xB - xA) * (yB - yA)
        box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = interArea / float(box1Area + box2Area - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou


def evaluate_one_ada(pred_boxes, truth_boxes_o):
    if not truth_boxes_o and not pred_boxes:
        return 1, 1, 1
    truth_boxes = copy.deepcopy(truth_boxes_o)
    tp = 0
    for pred_box in pred_boxes:
        for truth_box in truth_boxes:
            if iou(pred_box, truth_box) > 0.5:
                tp += 1
                truth_boxes.remove(truth_box)
                break
    accuracy = tp / (len(pred_boxes) + len(truth_boxes_o) + tp)
    try:
        precision = tp / (len(pred_boxes) + tp)
    except:
        precision = 0
    try:
        recall = tp / (len(truth_boxes_o) + tp)
    except:
        recall = 0

    return accuracy, precision, recall


def evaluate_one(pred_boxes, truth_boxes_o):
    truth_boxes = copy.deepcopy(truth_boxes_o)
    tp = 0
    for pred_box in pred_boxes:
        for truth_box in truth_boxes:
            if iou(pred_box, truth_box) > 0.5:
                tp += 1
                truth_boxes.remove(truth_box)
                break

    return tp, len(pred_boxes) - tp, len(truth_boxes_o) - tp


def evaluate_ada(pred_boxes, truth_boxes_o, domains, confss, conf_thres):
    accs = {}
    pres = {}
    recs = {}
    accuracies = []
    precisions = []
    recalls = []
    for pbs_list, tbs_list, domain, confs in zip(pred_boxes, truth_boxes_o, domains, confss):
        filtered_pbs_list = [pbs for index, pbs in enumerate(pbs_list) if confs[index] >= conf_thres]
        acc, pre, rec = evaluate_one_ada(filtered_pbs_list, tbs_list)
        if domain not in accs:
            accs[domain] = [acc]
        else:
            accs[domain].append(acc)
        if domain not in pres:
            pres[domain] = [pre]
        else:
            pres[domain].append(pre)
        if domain not in recs:
            recs[domain] = [rec]
        else:
            recs[domain].append(rec)
    for key in accs.keys():
        accuracies.append(sum(accs[key]) / len(accs[key]))
    for key in pres.keys():
        precisions.append(sum(pres[key]) / len(pres[key]))
    for key in recs.keys():
        recalls.append(sum(recs[key]) / len(recs[key]))
    acc = sum(accuracies) / len(accuracies)
    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    return acc, p, r


def evaluate(pred_boxes, truth_boxes_o, domains, confss, conf_thres):
    tps = {}
    fps = {}
    fns = {}
    accuracies = []
    precisions = []
    recalls = []
    for pbs_list, tbs_list, domain, confs in zip(pred_boxes, truth_boxes_o, domains, confss):
        filtered_pbs_list = [pbs for index, pbs in enumerate(pbs_list) if confs[index] >= conf_thres]
        tp, fp, fn = evaluate_one(filtered_pbs_list, tbs_list)
        if domain not in tps:
            tps[domain] = tp
        else:
            tps[domain] += tp
        if domain not in fps:
            fps[domain] = fp
        else:
            fps[domain] += fp
        if domain not in fns:
            fns[domain] = fn
        else:
            fns[domain] += fn
    all_keys = set(list(tps.keys()) + list(fps.keys()) + list(fns.keys()))
    for key in all_keys:
        try:
            accuracies.append(tps[key] / (tps[key] + fps[key] + fns[key]))
        except:
            accuracies.append(0)
        try:
            precisions.append(tps[key] / (tps[key] + fps[key]))
        except:
            precisions.append(0)
        try:
            recalls.append(tps[key] / (tps[key] + fns[key]))
        except:
            recalls.append(0)
    acc = sum(accuracies) / len(accuracies)
    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    return acc, p, r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True, type=str, help='ground truth .csv file path')
    parser.add_argument('--res', required=True, type=str, help='detection result folder')
    parser.add_argument('--ada', action='store_true', help='whether to use Average Domain Accuracy')
    args = parser.parse_args()

    # check paths
    assert os.path.exists(os.path.join(args.res, 'labels'))
    assert os.path.exists(os.path.join(args.res, 'names.txt'))
    assert os.path.exists(args.gt)

    # read ground truth
    truths = pd.read_csv(args.gt, index_col=0)

    # process each result
    best_acc = 0.0
    best_score_threshold = 0.0

    names = []
    with open(os.path.join(args.res, 'names.txt')) as f:
        for line in f:
            names.append(line.strip())

    gt_lst = []
    domain_lst = []

    def parse_boxes(boxes_string):
        if boxes_string == 'no_box':
            return []
        return [list(map(int, box.strip().split(' '))) for box in boxes_string.split(';')]

    for image_name in names:
        # meta
        entry = truths[truths['image_name'] == image_name].index[0]

        # groundtruth
        gt_str = truths['BoxesString'][entry]
        gts = parse_boxes(gt_str)
        gt_lst.append(gts)

        # domain
        domain = truths['domain'][entry]
        domain_lst.append(domain)

    for score_threshold in tqdm(np.arange(0, 1, 0.02), total=np.arange(0, 1, 0.02).shape[0]):
        pred_lst = []
        conf_lst = []

        for image_name in names:
            # prediction
            lp = os.path.join(args.res, 'labels', image_name + '.txt')
            preds = []
            confs = []
            if os.path.exists(lp):
                with open(lp) as f:
                    for l in f:
                        pred, conf = parse_box(l.strip())
                        preds.append(pred)
                        confs.append(conf)
            pred_lst.append(preds)
            conf_lst.append(confs)

        if args.ada:
            acc, p, r = evaluate_ada(pred_lst, gt_lst, domain_lst, conf_lst, score_threshold)
        else:
            acc, p, r = evaluate(pred_lst, gt_lst, domain_lst, conf_lst, score_threshold)
        if acc > best_acc:
            best_score_threshold = score_threshold
            best_acc = acc

    print(f'Best score threshold: {best_score_threshold}')
    print(f'The best acc is: {best_acc}')


if __name__ == "__main__":
    main()
