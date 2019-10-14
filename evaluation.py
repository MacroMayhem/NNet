__author__ = "Aditya Singh"
__version__ = "0.1"
import os
import json
import torch
from utils import get_train_loader, get_test_loader, get_network, evaluate, plot_accuracies
import numpy as np


root_path = '/home/aditya/PycharmProjects/NNet/checkpoint/cifar100/StepClasses-10/BufferSamples-5/resnet18_modx/2019-10-11T17:41:44.444391'
eval_path = os.path.join(root_path, 'Evaluation')
with open(os.path.join(root_path, 'setting.json')) as f:
    SETTINGS = json.load(f)

model_ckp = os.path.join(root_path, '{net}-{idx}-{epoch}-{type}.pth')

net = get_network(net=SETTINGS['NET'], num_classes=SETTINGS['STEP_CLASSES'], input_channels=3)

if not os.path.exists(eval_path):
    os.makedirs(eval_path)
    os.makedirs(eval_path+'/OldAccuracy', exist_ok=True)
    os.makedirs(eval_path+'/NewAccuracy', exist_ok=True)
    os.makedirs(eval_path+'/AvgIncrementalAccuracy', exist_ok=True)

incremental_accuracy = []
for iteration, sequence in enumerate(SETTINGS['TRAINING_BATCHES']):
    test_loader = get_test_loader(dataset=SETTINGS['DATASET'], accepted_class_labels=sequence, num_workers=0)
    cum_old_accuracies = [0]
    new_accuracies = []

    for epoch in range(SETTINGS['EPOCH']):
        net.load_state_dict(torch.load(model_ckp.format(net=SETTINGS['NET'], idx=iteration, epoch=epoch, type='end')))
        current_acc = evaluate(net, test_loader, label_correction=iteration*SETTINGS['STEP_CLASSES'])
        old_accuracies = []
        for old_iteration in range(iteration):
            old_sequence = SETTINGS['TRAINING_BATCHES'][old_iteration]
            old_test_loader = get_test_loader(dataset=SETTINGS['DATASET'], accepted_class_labels=old_sequence, num_workers=0)
            acc = evaluate(net, old_test_loader, label_correction=old_iteration*SETTINGS['STEP_CLASSES'])
            old_accuracies.append(acc.cpu().numpy())
        new_accuracies.append(current_acc.cpu().numpy())
        if iteration > 0:
            cum_old_accuracies.append(np.mean(np.asarray(old_accuracies)))
            plot_accuracies(old_accuracies, eval_path+'/OldAccuracy/{}'.format(iteration), '{}'.format(epoch))
    incremental_accuracy.append((new_accuracies[-1]+cum_old_accuracies[-1]*iteration)/(iteration+1))
