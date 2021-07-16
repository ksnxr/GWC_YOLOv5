import pandas as pd
import os
import argparse
from pathlib import Path
import copy
import time


def parse_box(line):
    _, x_center, y_center, width, height, _ = list(map(lambda x: float(x) * 1024, line.split(' ')))

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

    return f'{x_min} {y_min} {x_max} {y_max}'


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


def parse_boxes(boxes_string):
    if boxes_string == 'no_box':
        return []
    return [list(map(int, box.strip().split(' '))) for box in boxes_string.split(';')]


def evaluate_ada(pred_boxes, truth_boxes_o, domains):
    accs = {}
    pres = {}
    recs = {}
    accuracies = []
    precisions = []
    recalls = []
    for pbs_string, tbs_string, domain in zip(pred_boxes, truth_boxes_o, domains):
        pbs_list = parse_boxes(pbs_string)
        tbs_list = parse_boxes(tbs_string)
        acc, pre, rec = evaluate_one_ada(pbs_list, tbs_list)
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


def evaluate(pred_boxes, truth_boxes_o, domains):
    tps = {}
    fps = {}
    fns = {}
    accuracies = []
    precisions = []
    recalls = []
    for pbs_string, tbs_string, domain in zip(pred_boxes, truth_boxes_o, domains):
        pbs_list = parse_boxes(pbs_string)
        tbs_list = parse_boxes(tbs_string)
        tp, fp, fn = evaluate_one(pbs_list, tbs_list)
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
        accuracies.append(tps[key] / (tps[key] + fps[key] + fns[key]))
        precisions.append(tps[key] / (tps[key] + fps[key]))
        recalls.append(tps[key] / (tps[key] + fns[key]))
    acc = sum(accuracies) / len(accuracies)
    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    return acc, p, r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True, type=str, help='ground truth .csv file path')
    parser.add_argument('--res', nargs='+', type=str, help='detection result folders')
    parser.add_argument('--ada', action='store_true', help='whether to use Average Domain Accuracy')
    args = parser.parse_args()

    # check paths
    for folder_path in args.res:
        assert os.path.exists(os.path.join(folder_path, 'labels'))
        assert os.path.exists(os.path.join(folder_path, 'names.txt'))
    assert os.path.exists(args.gt)

    # read ground truth
    truths = pd.read_csv(args.gt, index_col=0)

    # process each result
    best_acc = 0
    best_acc_paths = []
    for res_path in args.res:
        print(f"Evaluating on {res_path}...")
        label_dir = os.path.join(res_path, 'labels')
        label_paths = [Path(os.path.join(label_dir, p)) for p in os.listdir(label_dir)]

        gt_lst = []
        pred_lst = []
        domain_lst = []

        names = []
        with open(os.path.join(res_path, 'names.txt')) as f:
            for l in f:
                names.append(l.strip())

        for image_name in names:
            # meta
            entry = truths[truths['image_name'] == image_name].index[0]

            # groundtruth
            gt_str = truths['BoxesString'][entry]
            gt_lst.append(gt_str)

            # prediction
            lp = os.path.join(res_path, 'labels', image_name + '.txt')
            if not os.path.exists(lp):
                pred_str = 'no_box'
            else:
                pred_str = []
                with open(lp) as f:
                    for l in f:
                        pred_str.append(parse_box(l.strip()))
                if not pred_str:
                    pred_str = 'no_box'
                else:
                    pred_str = ';'.join(pred_str)
            pred_lst.append(pred_str)

            # domain
            domain = truths['domain'][entry]
            domain_lst.append(domain)

        if args.ada:
            acc, p, r = evaluate_ada(pred_lst, gt_lst, domain_lst)
        else:
            acc, p, r = evaluate(pred_lst, gt_lst, domain_lst)
        if acc > best_acc:
            best_acc_paths = [res_path]
            best_acc = acc
        elif acc == best_acc:
            best_acc_paths.append(res_path)

        print()

        s = "----- Evaluation Report -----\n"
        s += f"Generated on {time.asctime( time.localtime(time.time()) )}\n"
        s += f'Result path "{os.path.basename(res_path)}"\n'
        s += f'Val set {len(names)} images\n'
        s += 'Accuracy %.3f\n' % acc
        s += 'Precision %.3f\n' % p
        s += 'Recall %.3f\n' % r
        s += '[END]\n'

        with open(os.path.join(res_path, "performance.txt"), 'w') as f:
            f.write(s)

        print(s)
        print()
        print(f'Best acc path(s): {best_acc_paths}')
        print(f'The best acc is: {best_acc}')
        print()


if __name__ == "__main__":
    main()
