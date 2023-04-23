import utils
import json
from operator import add
import numpy as np
from matplotlib import pyplot as plt

models = ["mlp_1","mlp_2","cnn_3","cnn_4","cnn_5"]

# results = [[],[],[],[],[]]
results = []
result = {"name": "mpty",
           "loss_curve": [],
           "train_acc_curve": [],
           "val_acc_curve": [],
           "test_acc": 0.0,
           "weights": []}


model_name = "mlp_1"
i = 0
for model_name in models:
    i += 1
    # print(model_name)
    # model_name = str(each)
    # exec("%s = %d" % (model_name +"_training_loss_record", 0))
    training_loss_record = []
    training_acc_record = []
    validation_loss_record = []
    validation_acc_record = []
    test_acc_record = []

    training_loss_record_average=[]
    training_acc_record_average=[]
    validation_loss_record_average=[]
    validation_acc_record_average=[]

    with open("./results/["+ model_name +']training_loss_record', "r") as fp:
        training_loss_record = json.load(fp)
    with open("./results/["+ model_name +']training_acc_record', "r") as fp:
        training_acc_record = json.load(fp)
    with open("./results/["+ model_name +']validation_loss_record', "r") as fp:
        validation_loss_record = json.load(fp)
    with open("./results/["+ model_name +']validation_acc_record', "r") as fp:
        validation_acc_record = json.load(fp)
    with open("./results/["+ model_name +']test_acc_record', "r") as fp:
        test_acc_record = json.load(fp)

    weights=np.load("./results/["+ model_name +']weights.npy')

    # for different runs compute the average loss
    training_loss_record_average = [ sum(x) for x in zip(*training_loss_record) ]
    training_loss_record_average = [ x/len(training_loss_record) for x in training_loss_record_average ]

    training_acc_record_average = [ sum(x) for x in zip(*training_acc_record) ]
    training_acc_record_average = [ x/len(training_acc_record) for x in training_acc_record_average ]

    validation_loss_record_average = [ sum(x) for x in zip(*validation_loss_record) ]
    validation_loss_record_average = [ x/len(validation_loss_record) for x in validation_loss_record_average ]

    validation_acc_record_average = [ sum(x) for x in zip(*validation_acc_record) ]
    validation_acc_record_average = [ x/len(validation_acc_record) for x in validation_acc_record_average ]

    test_acc_record_best = max(test_acc_record)

    result['name'] = model_name
    result['loss_curve'] = training_loss_record_average
    result['train_acc_curve'] = training_acc_record_average
    result['val_acc_curve'] = validation_acc_record_average
    result['test_acc'] = test_acc_record_best
    result['weights'] = weights.tolist()
    utils.visualizeWeights(weights, save_dir="./out/", filename=model_name+"_weights")


    results.append(result.copy())

# utils.part3Plots(results, save_dir="./results/",filename="aa",show_plot=True)








