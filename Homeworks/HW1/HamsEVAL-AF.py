import utils
import json
from operator import add
import numpy as np
from matplotlib import pyplot as plt

models = ["mlp_1[ReLU]", "mlp_1[Sigmoid]", "mlp_2[ReLU]", "mlp_2[Sigmoid]", "cnn_3[ReLU]", "cnn_3[Sigmoid]", "cnn_4[ReLU]", "cnn_4[Sigmoid]", "cnn_5[ReLU]", "cnn_5[Sigmoid]"]

results = []
result = {"name": "mpty",
           "relu_loss_curve": [],
           "sigmoid_loss_curve": [],
           "relu_grad_curve": [],
           "sigmoid_grad_curve": []}

i = 0 
for model_name in models:
    i += 1
    training_loss_record = []
    training_grad_record = []

    with open("./results/["+ model_name +']training_loss_record', "r") as fp:
        training_loss_record = json.load(fp)
    with open("./results/["+ model_name +']training_grad_record', "r") as fp:
        training_grad_record = json.load(fp)

    result['name'] = model_name[:5]
    if i % 2 == 1:
        results.append([])
    if model_name[6:13] == "Sigmoid":
        result['sigmoid_loss_curve'] = training_loss_record
        result['sigmoid_grad_curve'] = training_grad_record
        results[int((i-2)/2)]= result.copy()
    elif model_name[6:10] == "ReLU":
        result['relu_loss_curve'] = training_loss_record
        result['relu_grad_curve'] = training_grad_record
        results[int((i-1)/2)]=result.copy()
    print(i)

    

utils.part4Plots(results, save_dir="./results/", filename='bb', show_plot=True)
