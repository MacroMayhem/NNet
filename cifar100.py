from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import enum
from pathlib import Path
from utils import get_train_loader, get_test_loader, save_setting, plot_norm_losses, get_network, plot_embedding
from losses import norm_triangle

SETTINGS = {
    'CHECKPOINT_ROOT': './checkpoint',
    'EPOCH': 10,
    'TIME_NOW': str(datetime.now().isoformat()),
    'SAVE_EVERY': 1,
    'BATCH_SIZE': 128,
    'LEARNING_RATE': 0.0001,
    'NET': 'resnet18',
    'DATASET': 'cifar100',
    'STEP_CLASSES': 10,
    'NUM_CLASSES': 100,
    'NORM_LAMBDA': 1,
    'K_SHOT': 5
}


class MODE(enum.Enum):
    SUPER_DEBUG = 0
    DEBUG = 1
    NORMAL = 2

mode = MODE.DEBUG

def main():

    classes = [i for i in range(SETTINGS['NUM_CLASSES'])]
    training_batches = [classes[i:i + SETTINGS['STEP_CLASSES']] for i in range(0, len(classes), SETTINGS['STEP_CLASSES'])]
    SETTINGS['TRAINING_BATCHES'] = training_batches
    checkpoint_path = os.path.join(SETTINGS['CHECKPOINT_ROOT'], SETTINGS['DATASET'], 'StepClasses-{}'.format(str(SETTINGS['STEP_CLASSES']))
                                   , 'BufferSamples-{}'.format(str(SETTINGS['K_SHOT'])), SETTINGS['NET'], SETTINGS['TIME_NOW'])

    if not os.path.exists(checkpoint_path):
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    model_ckp_path = os.path.join(checkpoint_path, '{net}-{idx}-{epoch}-{type}.pth')
    save_setting(SETTINGS, checkpoint_path)

    if SETTINGS['NET'] == 'resnet18':
        net = get_network(net=SETTINGS['NET'], num_classes=SETTINGS['NUM_CLASSES'], input_channels=3)
    else:
        raise NotImplementedError('{} currently not supported!'.format(SETTINGS['NET']))

    norm_zero_loss = torch.nn.L1Loss()
    norm_alpha_loss = torch.nn.L1Loss()
    norm_triangle_loss = norm_triangle
    optimizer = optim.Adam(params=net.parameters(), lr=SETTINGS['LEARNING_RATE'])

    zero_img = torch.zeros(size=[1, 3, 32, 32])
    zero_label = torch.zeros(size=[1, 512])

    for iteration, training_sequence in enumerate(training_batches):
        if not os.path.exists(os.path.join(checkpoint_path, 'Plots', str(iteration))):
            Path(os.path.join(checkpoint_path, 'LossPlots', str(iteration))).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(checkpoint_path, 'EmbeddingPlots', str(iteration))).mkdir(parents=True, exist_ok=True)

        training_loader = get_train_loader(SETTINGS['DATASET'], accepted_class_labels=training_sequence, norm_lambda=SETTINGS['NORM_LAMBDA'], batch_size=SETTINGS['BATCH_SIZE'])
        test_loader = get_test_loader(SETTINGS['DATASET'], accepted_class_labels=training_sequence, batch_size=10*SETTINGS['BATCH_SIZE'])
        for epoch in range(SETTINGS['EPOCH']):
            print('Processing iteration: {}\nEpoch:{}'.format(iteration, epoch))
            for batch_idx, data in enumerate(training_loader):
                x, y, alpha, x2, y2, x_alpha, y_alpha = data
                if mode == MODE.SUPER_DEBUG:
                    print('---INPUT SHAPES---')
                    print(x.shape, y.shape, alpha.shape, x2.shape, y2.shape, x_alpha.shape, y_alpha.shape)

                net.eval()
                with torch.no_grad():
                    _, x_features = net(x.cuda())
                    _, x2_features = net(x2.cuda())
                    _, alpha_x_features = net(x_alpha.cuda())
                    #_, add_x_features = net((x+x2).cuda())
                #if mode == MODE.SUPER_DEBUG:
                    #print('---OUTPUT SHAPES---')
                    #print(x_features.shape, x2_features.shape, alpha_x_features.shape, add_x_features.shape)

                net.train()
                net.zero_grad()
                _, x_training_features = net(x.cuda())
                l_a = norm_alpha_loss(x_training_features*torch.unsqueeze(alpha, dim=1).cuda(), alpha_x_features)
                #l_t = norm_triangle_loss(x_features, add_x_features, x2_features)
                #agg_loss = l_a + l_t
                l_a.backward()
                optimizer.step()

                net.zero_grad()
                _, zero_features = net(zero_img.cuda())
                l_z = norm_zero_loss(zero_features, zero_label.cuda()) / torch.as_tensor(SETTINGS['BATCH_SIZE'])
                l_z.backward()
                optimizer.step()
                plot_norm_losses(l_a.item(), 0, l_z.item()
                                 , path=os.path.join(checkpoint_path, 'LossPlots', str(iteration))
                                 , fid='Epoch:{}--BatchNo:{}'.format(epoch, batch_idx))

                for data in test_loader:
                    x_test, y_test, _, _, _, _, _ = data
                    break
                net.eval()
                with torch.no_grad():
                    _, x_test_features = net(x_test.cuda())
                plot_embedding(x_test_features.cpu().numpy(), y_test.cpu().numpy(), num_classes=SETTINGS['STEP_CLASSES']
                               , filepath=os.path.join(checkpoint_path, 'EmbeddingPlots', str(iteration))
                               , filename='Epoch:{}--BatchNo:{}'.format(epoch, batch_idx))

            plot_norm_losses(l_a.item(), 0, l_z.item(), path=os.path.join(checkpoint_path, 'LossPlots',
                                                                 str(iteration)
                                                                 ), fid='Epoch:{}--END'.format(epoch))
        break


if __name__ == '__main__':
    main()