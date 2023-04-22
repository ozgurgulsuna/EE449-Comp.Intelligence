import utils
import json
from operator import add
from matplotlib import pyplot as plt

model_name = "mlp_1"

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

# for different runs compute the average loss
training_loss_record_average = [ sum(x) for x in zip(*training_loss_record) ]
training_loss_record_average = [ x/len(training_loss_record) for x in training_loss_record_average ]

training_acc_record_average = [ sum(x) for x in zip(*training_acc_record) ]
training_acc_record_average = [ x/len(training_acc_record) for x in training_acc_record_average ]

validation_loss_record_average = [ sum(x) for x in zip(*validation_loss_record) ]
validation_loss_record_average = [ x/len(validation_loss_record) for x in validation_loss_record_average ]

validation_acc_record_average = [ sum(x) for x in zip(*validation_acc_record) ]
validation_acc_record_average = [ x/len(validation_acc_record) for x in validation_acc_record_average ]






