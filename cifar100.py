from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import enum
from pathlib import Path
from utils import get_train_loader, get_test_loader, save_setting, plot_norm_losses, get_network, plot_embedding, plot_gradients
from losses import norm_triangle, zero_loss

SETTINGS = {
    'CHECKPOINT_ROOT': './checkpoint',
    'STARTING_EPOCH': 50,
    'OTHER_EPOCHS': 20,
    'TIME_NOW': str(datetime.now().isoformat()),
    'SAVE_EVERY': 1,
    'BATCH_SIZE': 128,
    'STARTING_LEARNING_RATE': 0.1, #Initial Learning Rate
    'OTHER_LEARNING_RATE': 0.01,
    'NET': 'resnet18_modx',
    'DATASET': 'cifar100',
    'STEP_CLASSES': 10,
    'NUM_CLASSES': 100,
    'NORM_LAMBDA': 1,
    'K_SHOT': 5,
    'LOSSES':['CE'],
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

    net = get_network(net=SETTINGS['NET'], num_classes=SETTINGS['STEP_CLASSES'], input_channels=3)

    norm_alpha_loss = torch.nn.MSELoss()
    norm_triangle_loss = torch.nn.MSELoss()
    ce_criterion = torch.nn.CrossEntropyLoss()
    zero_img = torch.zeros(size=[1, 3, 32, 32])
    zero_label = torch.zeros(size=[1, 512])

    old_classes = []

    for iteration, training_sequence in enumerate(training_batches):
        if not os.path.exists(os.path.join(checkpoint_path, 'Plots', str(iteration))):
            base_path = os.path.join(checkpoint_path, 'Plots', str(iteration))
            base_gradients_path = os.path.join(base_path, 'Gradients')
            g_zero_path = os.path.join(base_gradients_path, 'L_Zero')
            g_alpha_path = os.path.join(base_gradients_path, 'L_Alpha')
            g_triangle_path = os.path.join(base_gradients_path, 'L_Triangle')
            loss_path = os.path.join(base_path, 'LossPlots')
            embedding_path = os.path.join(base_path, 'EmbeddingPlots')
            Path(loss_path).mkdir(parents=True, exist_ok=True)
            Path(embedding_path).mkdir(parents=True, exist_ok=True)
            Path(g_zero_path).mkdir(parents=True, exist_ok=True)
            Path(g_alpha_path).mkdir(parents=True, exist_ok=True)
            Path(g_triangle_path).mkdir(parents=True, exist_ok=True)

        training_loader = get_train_loader(SETTINGS['DATASET'], accepted_class_labels=training_sequence, norm_lambda=SETTINGS['NORM_LAMBDA'], batch_size=SETTINGS['BATCH_SIZE'])
        old_classes.extend(training_sequence)
        test_loader = get_test_loader(SETTINGS['DATASET'], accepted_class_labels=old_classes, batch_size=5*SETTINGS['BATCH_SIZE'])

        if iteration == 0:
            EPOCH = SETTINGS['OTHER_EPOCHS']
            lr = SETTINGS['STARTING_LEARNING_RATE']
        else:
            EPOCH = SETTINGS['STARTING_EPOCH']
            lr = SETTINGS['OTHER_LEARNING_RATE']

        ce_optimizer = optim.SGD(params=net.parameters(), lr=lr, momentum=0.9)
        triangle_optimizer = optim.SGD(params=net.parameters(), lr=lr, momentum=0.9)
        zero_optimizer = optim.SGD(params=net.parameters(), lr=lr, momentum=0.9)

        for epoch in range(EPOCH):
            print('Processing iteration: {}\nEpoch:{}'.format(iteration, epoch))
            for batch_idx, data in enumerate(training_loader):
                x, y, alpha, x2, y2, x_alpha, x_convex = data
                y = y - iteration*SETTINGS['STEP_CLASSES']

                if mode == MODE.SUPER_DEBUG:
                    print('---INPUT SHAPES---')
                    print(x.shape, y.shape, alpha.shape, x2.shape, y2.shape, x_alpha.shape, x_convex.shape)

                net.eval()
                with torch.no_grad():
                    _, x2_features = net(x2.cuda())
                    _, alpha_x_features = net(x_alpha.cuda())

                alpha_sq = torch.unsqueeze(alpha, dim=1)
                if 'CE' in SETTINGS['LOSSES']:
                    net.train()
                    net.zero_grad()
                    preds, x_features = net(x.cuda())
                    l_ce = ce_criterion(preds, y.cuda())
                    l_ce.backward(retain_graph=True)
                    ce_gradients = get_gradient_magnitudes(net)
                    plot_gradients(ce_gradients, g_alpha_path, '{}--{}'.format(epoch, batch_idx))
                    del ce_gradients
                    ce_optimizer.step()
                else:
                    l_ce = DummyLoss()

                """net.train()
                net.zero_grad()
                _, x_features = net(x.cuda())
                x_norm = torch.unsqueeze(torch.norm(x_features, p=2, dim=1), dim=1)
                alpha_sq = torch.unsqueeze(alpha, dim=1)
                alpha_x_norm = torch.unsqueeze(torch.norm(alpha_x_features, p=2, dim=1), dim=1)
                # print(alpha_sq.shape, x_norm.shape, alpha_x_norm.shape)
                l_a = norm_alpha_loss(x_norm*alpha_sq.cuda(), alpha_x_norm)
                l_a.backward(retain_graph=True)
                alpha_gradients = get_gradient_magnitudes(net)
                plot_gradients(alpha_gradients, g_alpha_path, '{}--{}'.format(epoch, batch_idx))
                del alpha_gradients
                ce_optimizer.step()"""

                if 'TRIANGLE' in SETTINGS['LOSSES']:
                    net.train()
                    net.zero_grad()
                    _, cvx_features = net(x_convex.cuda())
                    l_t = norm_triangle_loss(torch.log(torch.unsqueeze(torch.norm(cvx_features, p=2, dim=1), dim=1))
                         , torch.log(alpha_sq.cuda() * torch.unsqueeze(torch.norm(x_features, p=2, dim=1), dim=1)
                         + (1 - alpha_sq.cuda()) * torch.unsqueeze(torch.norm(x2_features, p=2, dim=1), dim=1)))
                    l_t.backward()
                    triangle_gradients = get_gradient_magnitudes(net)
                    plot_gradients(triangle_gradients, g_triangle_path, '{}--{}'.format(epoch, batch_idx))
                    del triangle_gradients
                    triangle_optimizer.step()
                else:
                    l_t = DummyLoss()


                """net.zero_grad()
                _, zero_features = net(zero_img.cuda())
                l_z = zero_loss(zero_features)/SETTINGS['BATCH_SIZE']
                l_z.backward()
                zero_gradients = get_gradient_magnitudes(net)
                plot_gradients(zero_gradients, g_zero_path, '{}--{}'.format(epoch, batch_idx))
                del zero_gradients
                zero_optimizer.step()"""
                plot_norm_losses(l_ce.item(), l_t.item(), 0
                                 , path=loss_path
                                 , fid='Epoch:{}--BatchNo:{}'.format(epoch, batch_idx))

            train_acc = evaluate(net, training_loader, label_correction=iteration*SETTINGS['STEP_CLASSES'])
            print('Training accuracy: {}'.format(train_acc))
            test_features = None
            test_labels = None
            for data in test_loader:
                x_test, y_test, _, _, _, _, _ = data
                net.eval()
                with torch.no_grad():
                    _, x_test_features = net(x_test.cuda())
                    if test_features is None:
                        test_features = x_test_features.cpu()
                        test_labels = y_test.cpu()
                    else:
                        test_features = torch.cat([test_features, x_test_features.cpu()], dim=0)
                        test_labels = torch.cat([test_labels, y_test.cpu()], dim=0)
            plot_embedding(test_features.numpy(), test_labels.numpy(), num_classes=len(old_classes)
                           , filepath=embedding_path
                           , filename='Epoch:{}'.format(epoch))
            torch.save(net.state_dict(),
                       model_ckp_path.format(net=SETTINGS['NET'], idx=iteration, epoch=epoch, type='end'))


def get_gradient_magnitudes(net):
    gradient_magnitudes = []
    for name, parameter in net.named_parameters():
        try:
            gradient_magnitudes.append(parameter.grad.norm(2).item() ** 2)
        except:
            continue
    return gradient_magnitudes


class DummyLoss:
    def __init__(self):
        self.val = 0.0
    def item(self):
        return self.val


def evaluate(net, dataloader, label_correction):
    net.eval()
    correct = 0.0
    with torch.no_grad():
        for data in dataloader:
            x, y, _, _, _, _, _ = data

            outputs, _ = net(x.cuda())
            _, preds = outputs.max(1)
            correct += preds.eq((y-label_correction).cuda()).sum()

    return correct.float() / len(dataloader.dataset)


if __name__ == '__main__':
    main()
