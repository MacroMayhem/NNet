from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import enum
from pathlib import Path
from utils import get_train_loader, get_test_loader, save_setting, plot_norm_losses, get_network, plot_embedding
from losses import norm_triangle, pos_loss, zero_loss

SETTINGS = {
    'CHECKPOINT_ROOT': './checkpoint',
    'EPOCH': 100,
    'TIME_NOW': str(datetime.now().isoformat()),
    'SAVE_EVERY': 1,
    'BATCH_SIZE': 128,
    'LEARNING_RATE': 0.0001,
    'NET': 'resnet18',
    'DATASET': 'cifar',
    'STEP_CLASSES': 10,
    'NUM_CLASSES': 100,
    'NORM_LAMBDA': 10,
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

    norm_alpha_loss = torch.nn.L1Loss()
    norm_triangle_loss = norm_triangle
    optimizer = optim.SGD(params=net.parameters(), lr=SETTINGS['LEARNING_RATE'], momentum=0.9)

    zero_img = torch.zeros(size=[1, 3, 32, 32])
    zero_label = torch.zeros(size=[1, 1])

    for iteration, training_sequence in enumerate(training_batches):
        if not os.path.exists(os.path.join(checkpoint_path, 'Plots', str(iteration))):
            Path(os.path.join(checkpoint_path, 'LossPlots', str(iteration))).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(checkpoint_path, 'EmbeddingPlots', str(iteration))).mkdir(parents=True, exist_ok=True)

        training_loader = get_train_loader(SETTINGS['DATASET'], accepted_class_labels=training_sequence, norm_lambda=SETTINGS['NORM_LAMBDA'], batch_size=SETTINGS['BATCH_SIZE'])
        test_loader = get_test_loader(SETTINGS['DATASET'], accepted_class_labels=training_sequence, batch_size=5*SETTINGS['BATCH_SIZE'])
        for epoch in range(SETTINGS['EPOCH']):
            print('Processing iteration: {}\nEpoch:{}'.format(iteration, epoch))
            for batch_idx, data in enumerate(training_loader):
                x, y, alpha, x2, y2, x_alpha, y_alpha = data
                if mode == MODE.SUPER_DEBUG:
                    print('---INPUT SHAPES---')
                    print(x.shape, y.shape, alpha.shape, x2.shape, y2.shape, x_alpha.shape, y_alpha.shape)

                net.eval()
                with torch.no_grad():
                    #_, _ = net(x.cuda())
                    n_x2, _ = net(x2.cuda())
                    n_x_alpha, _ = net(x_alpha.cuda())
                    n_x_add_x2, _ = net((x+x2).cuda())

                net.train()
                net.zero_grad()
                n_x, _ = net(x.cuda())
                l_a = norm_alpha_loss(n_x*torch.unsqueeze(alpha, dim=1).cuda(), n_x_alpha)

                l_t = norm_triangle_loss(n_x, n_x_add_x2, n_x2)
                l_p = pos_loss(n_x)
                agg_loss = l_t+l_p
                agg_loss.backward()
                optimizer.step()

                net.zero_grad()
                n_zero, _ = net(zero_img.cuda())
                l_z = zero_loss(n_zero)/SETTINGS['BATCH_SIZE']
                l_z.backward()
                optimizer.step()
                plot_norm_losses(0, l_t.item(), l_z.item(), l_p.item()
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

            plot_norm_losses(l_a.item(), l_t.item(), l_z.item(), l_p.item(), path=os.path.join(checkpoint_path, 'LossPlots',
                                                                            str(iteration)), fid='Epoch:{}--END'.format(epoch))
        break


if __name__ == '__main__':
    main()