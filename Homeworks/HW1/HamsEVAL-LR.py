import utils
import json
from operator import add
import numpy as np
from matplotlib import pyplot as plt

model_name = 'cnn_4'

results = []
result = {"name": "mpty",
           "loss_curve_1": [],
           "loss_curve_01": [],
           "loss_curve_001": [],
           "val_acc_curve_1": [],
           "val_acc_curve_01": [],
           "val_acc_curve_001": []}

with open("./results/["+ model_name +']RL1_training_loss_record', "r") as fp:
    RL1_training_loss_record = json.load(fp)
with open("./results/["+ model_name +']RL1_validation_acc_record', "r") as fp:
    RL1_validation_acc_record = json.load(fp)
with open("./results/["+ model_name +']RL01_training_loss_record', "r") as fp:
    RL01_training_loss_record = json.load(fp)
with open("./results/["+ model_name +']RL01_validation_acc_record', "r") as fp:
    RL01_validation_acc_record = json.load(fp)
with open("./results/["+ model_name +']RL001_training_loss_record', "r") as fp:
    RL001_training_loss_record = json.load(fp)
with open("./results/["+ model_name +']RL001_validation_acc_record', "r") as fp:
    RL001_validation_acc_record = json.load(fp)

result['name'] = model_name[:5]
result['loss_curve_1'] = RL1_training_loss_record
result['loss_curve_01'] = RL01_training_loss_record
result['loss_curve_001'] = RL001_training_loss_record
result['val_acc_curve_1'] = RL1_validation_acc_record
result['val_acc_curve_01'] = RL01_validation_acc_record
result['val_acc_curve_001'] = RL001_validation_acc_record

utils.part5Plots(result, save_dir="./results/", filename='cc', show_plot=True)
