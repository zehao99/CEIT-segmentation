import torch


def get_accuracy(SR, GT, threshold=0.5):
    """
    Get the accuracy between Ground Truth and Segmentation Results

    Args:
        SR: tensor  Segmentation Results
        GT: tensor  Ground Truth
        threshold: the threshold of segmentation

    Returns:
        acc: accuracy between SR and GT
    """
    SR = SR > threshold
    GT = GT > 0
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    acc = float(corr) / float(tensor_size)
    return acc


def get_recall(SR, GT, threshold=0.5):
    """

    Args:
        SR: tensor  Segmentation Results
        GT: tensor  Ground Truth
        threshold: the threshold of segmentation

    Returns:
        RC: Recall rate
    """
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = (SR == True) & (GT == True)
    FN = (SR == False) & (GT == True)

    RC = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return RC


def get_precision(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = (SR == True) & (GT == True)
    FP = (SR == True) & (GT == False)
    sum1 = float(torch.sum(TP))
    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    RC = get_recall(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * RC * PC / (RC + PC + 1e-6)

    return F1
