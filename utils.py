import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def calc_block_idx(x_min, x_max, y_min, y_max, h_step, w_step, mode):
    all_blocks = list()
    center = np.array([(y_min + y_max) / 2, (x_min + x_max) / 2])
    all_blocks.append(center + center)
    if mode > 1:
        all_blocks.append(np.array([y_min, center[1]]) + center)
        all_blocks.append(np.array([y_max, center[1]]) + center)
        all_blocks.append(np.array([center[0], x_min]) + center)
        all_blocks.append(np.array([center[0], x_max]) + center)
    if mode >= 9:
        all_blocks.append(np.array([y_min, x_min]) + center)
        all_blocks.append(np.array([y_max, x_max]) + center)
        all_blocks.append(np.array([y_max, x_min]) + center)
        all_blocks.append(np.array([y_min, x_max]) + center)
    all_blocks = np.array(all_blocks) / 2
    h_block_idxes = all_blocks[:, 0] / h_step
    w_block_idxes = all_blocks[:, 1] / w_step
    h_block_idxes, w_block_idxes = list(h_block_idxes.astype(np.int)), list(w_block_idxes.astype(np.int))
    # delete repeated elements
    all_blocks = set([x for x in zip(h_block_idxes, w_block_idxes)])
    all_blocks = [x for x in all_blocks]
    return all_blocks


def save_roc_pr_curve_data(scores, labels, file_path, verbose=True):
    scores = scores.flatten()
    labels = labels.flatten()

    scores_pos = scores[labels == 1]
    scores_neg = scores[labels != 1]

    truth = np.concatenate((np.zeros_like(scores_neg), np.ones_like(scores_pos)))
    preds = np.concatenate((scores_neg, scores_pos))
    fpr, tpr, roc_thresholds = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)

    # calculate EER
    fnr = 1 - tpr
    eer1 = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    eer2 = fnr[np.nanargmin(np.absolute(fnr - fpr))]

    # pr curve where "normal" is the positive class
    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(truth, preds)
    pr_auc_norm = auc(recall_norm, precision_norm)

    # pr curve where "anomaly" is the positive class
    precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(truth, -preds, pos_label=0)
    pr_auc_anom = auc(recall_anom, precision_anom)

    if verbose is True:
        print('AUC@ROC is {}'.format(roc_auc), 'EER1 is {}'.format(eer1), 'EER2 is {}'.format(eer2))

    np.savez_compressed(file_path,
                        preds=preds, truth=truth,
                        fpr=fpr, tpr=tpr, roc_thresholds=roc_thresholds, roc_auc=roc_auc,
                        precision_norm=precision_norm, recall_norm=recall_norm,
                        pr_thresholds_norm=pr_thresholds_norm, pr_auc_norm=pr_auc_norm,
                        precision_anom=precision_anom, recall_anom=recall_anom,
                        pr_thresholds_anom=pr_thresholds_anom, pr_auc_anom=pr_auc_anom)

    return roc_auc


