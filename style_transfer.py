import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_path_test', type=str, help='path to the test image(s)')
parser.add_argument('style_path_test', type=str, help='path to the test style(s)')
parser.add_argument('path_to_weight_files', type=str, help='path to the pretrained weight files of all decoders')
parser.add_argument('--gpu', type=int, default=0, help='GPU number to use')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
import neuralnet as nn
from theano import tensor as T
from scipy import misc
from net import VGG19, Decoder, StyleTransfer, indices
from load_data import prep_image_test

input_path_test = args.input_path_test
style_path_test = args.style_path_test

#dummy shape
input_shape = (None, 3, 224, 224)


def style_transfer():
    enc = VGG19(input_shape)
    decs = [Decoder(enc, i, name='decoder %d' % i) for i in indices]
    sty_net = StyleTransfer(enc, decs)

    X = T.tensor4('input')
    Y = T.tensor4('style')

    mon = nn.Monitor(current_folder=args.path_to_weight_files)
    for idx, layer in enumerate(indices):
        weights = mon.load('decoder-%d-final.npz' % layer)
        nn.utils.numpy2shared(weights, decs[idx].params)

    nn.set_training_off()
    X_styled = sty_net(X, Y)
    transfer = nn.function([X, Y], X_styled, name='transfer style')

    if os.path.isfile(input_path_test) and os.path.isfile(style_path_test):
        input = prep_image_test(misc.imread(input_path_test))
        style = prep_image_test(misc.imread(style_path_test))
        output = transfer(input, style)
        mon.imwrite('test %s' % input_path_test[:-4], input)
        mon.imwrite('test %s' % style_path_test[:-4], style)
        mon.imwrite('test %s %s' % (input_path_test[:-4], style_path_test[:-4]), output)
    elif os.path.isfile(input_path_test) and os.path.isdir(style_path_test):
        input = prep_image_test(misc.imread(input_path_test))
        style_files = os.listdir(style_path_test)
        for style_file in style_files:
            style = prep_image_test(misc.imread(os.path.join(style_path_test, style_file)))
            output = transfer(input, style)
            mon.imwrite('test %s' % style_file[:-4], style)
            mon.imwrite('test %s %s' % (input_path_test[:-4], style_file[:-4]), output)
        mon.imwrite('test %s' % input_path_test[:-4], input)
    elif os.path.isdir(input_path_test) and os.path.isfile(style_path_test):
        style = prep_image_test(misc.imread(style_path_test))
        input_files = os.listdir(input_path_test)
        for input_file in input_files:
            input = prep_image_test(misc.imread(os.path.join(input_path_test, input_file)))
            output = transfer(input, style)
            mon.imwrite('test %s' % input_file[:-4], input)
            mon.imwrite('test %s %s' % (input_file[:-4], style_path_test[:-4]), output)
        mon.imwrite('test %s' % style_path_test[:-4], style)
    else:
        style_files = os.listdir(style_path_test)
        input_files = os.listdir(input_path_test)
        for style_file in style_files:
            style = prep_image_test(misc.imread(os.path.join(style_path_test, style_file)))
            for input_file in input_files:
                input = prep_image_test(misc.imread(os.path.join(input_path_test, input_file)))
                output = transfer(input, style)
                mon.imwrite('test %s' % input_file[:-4], input)
                mon.imwrite('test %s %s' % (input_file[:-4], style_file[:-4]), output)
            mon.imwrite('test %s' % style_file[:-4], style)
    mon.flush()
    print('Testing finished!')


if __name__ == '__main__':
    style_transfer()
