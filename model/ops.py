import numpy as np
import cv2
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score


####################################################
### Regular operations
####################################################

def save_img(img, img_index, root_path, img_name, mode = "image"):
    img = np.uint8(255 * img)
    if mode == "image":            
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif mode == "heatmap":
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    img_path = os.path.join(root_path, str(img_index) + img_name)
    cv2.imwrite(img_path, img)


def get_accuracy(preds, labels):
    """
    Overall accuracy
    """
    preds = np.array(preds)
    labels = np.array(labels)
    accuracy = round(accuracy_score(y_true = labels, y_pred = preds), 5)
    """
    Per_class_recall
    """
    matrix = confusion_matrix(y_true = labels, y_pred = preds)
    print ("confusion_matrix:", matrix)
    recalls = matrix.diagonal().astype('float')/matrix.sum(axis = 1)

    normal_recall = round(recalls[0], 5)
    bleed_recall = round(recalls[1], 5)
    inflam_recall = round(recalls[2], 5)

    """
    Cohen kappa
    """  
    kappa = round(cohen_kappa_score(y1 = preds, y2 = labels), 5)


    return accuracy, normal_recall, bleed_recall, inflam_recall, kappa
