from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import torchvision.models as models
from pathlib import Path
from utils import get_train_loader, get_test_loader, save_setting, plot_norm_losses
from losses import norm_triangle

SETTINGS = {
    'CHECKPOINT_ROOT': './checkpoint',
    'EPOCH': 10,
    'TIME_NOW': str(datetime.now().isoformat()),
    'SAVE_EVERY': 1,
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 0.001,
    'NET': 'resnet18',
    'DATASET': 'mnist',
    'STEP_CLASSES': 2,
    'NUM_CLASSES': 10,
    'NORM_LAMBDA': 5.0,
    'K_SHOT': 5
}


def main():

    classes = [i for i in range(SETTINGS['NUM_CLASSES'])]
    training_batches = [classes[i:i + SETTINGS['STEP_CLASSES']] for i in range(0, len(classes), SETTINGS['STEP_CLASSES'])]
    SETTINGS['TRAINING_BATCHES'] = training_batches
    checkpoint_path = os.path.join(SETTINGS['CHECKPOINT_ROOT'], SETTINGS['DATASET'], 'StepClasses-{}'.format(str(SETTINGS['STEP_CLASSES']))
                                   , 'StepClasses-{}'.format(str(SETTINGS['K_SHOT'])), SETTINGS['NET'], SETTINGS['TIME_NOW'])

    if not os.path.exists(checkpoint_path):
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    model_ckp_path = os.path.join(checkpoint_path, '{net}-{idx}-{epoch}-{type}.pth')
    save_setting(SETTINGS, checkpoint_path)

    if SETTINGS['NET'] == 'resnet18':
        net = models.resnet18(pretrained=False, num_classes=SETTINGS['NUM_CLASSES'])
    else:
        raise NotImplementedError('{} currently not supported!'.format(SETTINGS['NET']))

    norm_zero_loss = torch.nn.MSELoss()
    norm_alpha_loss = torch.nn.MSELoss()
    norm_triangle_loss = norm_triangle
    optimizer = optim.SGD(params=net.parameters(), lr=SETTINGS['LEARNING_RATE'], momentum=0.9, weight_decay=5e-4)

    zero_img = torch.zeros(size=[1, 1, 28, 28])
    zero_label = torch.zeros(size=[1, 512])

    for iteration, training_sequence in enumerate(training_batches):
        if not os.path.exists(os.path.join(checkpoint_path, 'Plots', str(iteration))):
            Path(os.path.join(checkpoint_path, 'Plots', str(iteration))).mkdir(parents=True, exist_ok=True)

        training_loader = get_train_loader(SETTINGS['DATASET'], accepted_class_labels=training_sequence, norm_lambda=SETTINGS['NORM_LAMBDA'])
        test_loader = get_test_loader(SETTINGS['DATASET'], accepted_class_labels=training_sequence)
        total_loss_a = 0.0
        total_loss_t = 0.0
        for batch_idx, data in enumerate(training_loader):
            x, y, alpha, x2, y2 = data
            print('---INPUT SHAPES---')
            print(x.shape, y.shape, alpha.shape, x2.shape, y2.shape)

            net.eval()
            with torch.no_grad():
                _, x_features = net(x.cuda())
                _, x2_features = net(x2.cuda())
                _, alpha_x_features = net((x*alpha).cuda())
                _, add_x_features = net((x+x2).cuda())
            print('---OUTPUT SHAPES---')
            print(x_features.shape, x2_features.shape, alpha_x_features.shape, add_x_features.shape)

            net.train()
            net.zero_grad()
            _, x_training_features = net(x.cuda())
            l_a = norm_alpha_loss(alpha*x_training_features, alpha_x_features)
            l_t = norm_triangle_loss(x_features, add_x_features, x2_features)
            total_loss_a += l_a.item()
            total_loss_t += l_t.item()
            agg_loss = l_a + l_t
            agg_loss.backward()
            optimizer.step()
            break
        net.train()
        net.zero_grad()
        _, zero_features = net(zero_img.cuda())
        l_z = norm_zero_loss(zero_features, zero_label)
        l_z.backward()
        optimizer.step()

        plot_norm_losses(total_loss_a, total_loss_t, l_z, os.path.join(checkpoint_path, 'Plots', str(iteration)))


if __name__ == '__main__':
    main()