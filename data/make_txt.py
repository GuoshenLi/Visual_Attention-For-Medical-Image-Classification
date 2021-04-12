from torch.utils.data import Dataset
import csv
import os
import glob
import random
from PIL import Image
import torchvision.transforms as transforms
import tensorflow as tf

def preprocess_WCE_data(root_dir):

    ########################### train #############################
    normal = glob.glob(os.path.join(root_dir, 'training', 'normal', '*.jpg'))
    normal.sort()
    inflammatory = glob.glob(os.path.join(root_dir, 'training', 'inflammatory', '*.jpg'))
    inflammatory.sort()
    vascular = glob.glob(os.path.join(root_dir, 'training', 'vascularlesions', '*.jpg'))
    vascular.sort()

    with open('train.csv', 'wt', newline = '') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(normal)):
            filename = normal[k]
            writer.writerow([filename] + ['0'])

        for k in range(len(inflammatory)):
            filename = inflammatory[k]
            writer.writerow([filename] + ['1'])

        for k in range(len(vascular)):
            filename = vascular[k]
            writer.writerow([filename] + ['2'])



    with open('train.csv', newline='') as file:
        reader = csv.reader(file, delimiter=',')
        pairs = [row for row in reader]

    random.shuffle(pairs)

    with open('train_shuffle.csv', 'wt',newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in pairs:
            writer.writerow([k[0]] + [k[1]])


    ########################### test.csv #############################

    normal = glob.glob(os.path.join(root_dir, 'testing', 'normal', '*.jpg'))
    normal.sort()
    inflammatory = glob.glob(os.path.join(root_dir, 'testing', 'inflammatory', '*.jpg'))
    inflammatory.sort()
    vascular = glob.glob(os.path.join(root_dir, 'testing', 'vascularlesions', '*.jpg'))
    vascular.sort()

    with open('test.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        for k in range(len(normal)):
            filename = normal[k]
            writer.writerow([filename] + ['0'])

        for k in range(len(inflammatory)):
            filename = inflammatory[k]
            writer.writerow([filename] + ['1'])

        for k in range(len(vascular)):
            filename = vascular[k]
            writer.writerow([filename] + ['2'])


preprocess_WCE_data('WCE_data_warped')