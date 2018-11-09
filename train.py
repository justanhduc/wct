import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_path_train', type=str, help='path to the MS COCO train images')
parser.add_argument('style_path_train', type=str, help='path to the Wikiart train dataset')
parser.add_argument('input_path_val', type=str, default=None, help='path to the MS COCO val images')
parser.add_argument('style_path_val', type=str, default=None, help='path to the Wikiart val images')
parser.add_argument('--num_val_imgs', type=int, default=24, help='number of images used for validation during training')
parser.add_argument('--input_size', type=int, default=256, help='the size of input images. input images will '
                                                                'be cropped to this square size')
parser.add_argument('--bs', type=int, default=8, help='batch size')
parser.add_argument('--style_weight', type=float, default=1., help='weight for style loss')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0., help='decay rate for learning rate')
parser.add_argument('--n_epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--print_freq', type=int, default=100, help='frequency to show losses')
parser.add_argument('--valid_freq', type=int, default=500, help='frequency to validate the network in training')
parser.add_argument('--gpu', type=int, default=0, help='GPU number to use')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
import neuralnet as nn
from theano import tensor as T
import numpy as np
from net import VGG19, Decoder, StyleTransfer, indices
from load_data import DataManager, DataManagerStyleTransfer

input_path_train = args.input_path_train
input_path_val = args.input_path_val
style_path_train = args.style_path_train
style_path_val = args.style_path_val
num_val_imgs = args.num_val_imgs
input_shape = (None,) + (3, args.input_size, args.input_size)
bs = args.bs
weight = args.style_weight
lr = args.lr
decay = args.lr_decay
n_epochs = args.n_epochs
print_freq = args.print_freq
val_freq = args.valid_freq


def train():
    enc = VGG19(input_shape)
    decs = [Decoder(enc, i, name='decoder %d' % i) for i in indices]
    sty_net = StyleTransfer(enc, decs)

    X = T.tensor4('input')
    Y = T.tensor4('style')
    idx = T.scalar('iter', 'int32')
    X_ = nn.placeholder((bs,) + input_shape[1:], name='input_plhd')
    Y_ = nn.placeholder((bs,) + input_shape[1:], name='style_plhd')
    lr_ = nn.placeholder(value=lr, name='lr_plhd')

    nn.set_training_on()
    losses = [dec.cost(X) for dec in decs]
    updates = [nn.adam(loss[0] + weight * loss[1], dec.trainable, lr) for loss, dec in zip(losses, decs)]
    nn.anneal_learning_rate(lr_, idx, 'inverse', decay=decay)
    trains = [nn.function([], [loss[0], loss[1], dec(X, True)], givens={X: X_}, updates=update, name='train decoder')
              for loss, dec, update in zip(losses, decs, updates)]

    nn.set_training_off()
    X_styled = sty_net(X, Y)
    transfer = nn.function([], X_styled, givens={X: X_, Y: Y_}, name='transfer style')

    data_train = DataManager(X_, input_path_train, bs, n_epochs, True, num_val_imgs=num_val_imgs, input_shape=input_shape)
    data_test = DataManagerStyleTransfer((X_, Y_), (input_path_val, style_path_val), bs, 1, input_shape=input_shape)
    mon = nn.Monitor(model_name='WCT', valid_freq=print_freq)

    print('Training...')
    for it in data_train:
        results = [train(it) for train in trains]

        with mon:
            for layer, res in zip(indices, results):
                if np.isnan(res[0] + res[1]) or np.isinf(res[0] + res[1]):
                    raise ValueError('Training failed!')
                mon.plot('pixel loss at layer %d' % layer, res[0])
                mon.plot('feature loss at layer %d' % layer, res[1])

            if it % val_freq == 0:
                mon.imwrite('recon img at layer %d' % layer, res[2])

                for i in data_test:
                    img = transfer()
                    mon.imwrite('stylized image %d' % i, img)
                    mon.imwrite('input %d' % i, X_.get_value())
                    mon.imwrite('style %d' % i, Y_.get_value())

                for idx, dec in zip(indices, decs):
                    mon.dump(nn.utils.shared2numpy(dec.params), 'decoder-%d.npz' % idx, 5)
    mon.flush()
    for idx, dec in zip(indices, decs):
        mon.dump(nn.utils.shared2numpy(dec.params), 'decoder-%d-final.npz' % idx)
    print('Training finished!')


if __name__ == '__main__':
    train()
