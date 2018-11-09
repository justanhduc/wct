import numpy as np
import neuralnet as nn
import theano
from theano import tensor as T
import h5py

indices = [1, 4, 7, 12, 17]


def prep(x):
    conv = nn.Conv2DLayer((1, 3, 224, 224), 3, 1, no_bias=False, activation='linear', filter_flip=False,
                          border_mode='valid')
    kern = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]], 'float32')[:, :, None, None]
    conv.W.set_value(kern)
    conv.b.set_value(np.array([-103.939, -116.779, -123.68], 'float32'))
    return conv(x)


class VGG19(nn.Sequential):
    def __init__(self, input_shape, name='vgg19'):
        super(VGG19, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(
            nn.Conv2DLayer(self.output_shape, 64, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv1_1'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 64, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv1_2'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool1'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 128, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv2_1'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 128, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv2_2'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool2'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv3_1'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv3_2'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv3_3'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv3_4'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool3'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 512, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv4_1'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 512, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv4_2'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 512, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv4_3'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 512, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv4_4'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool4'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 512, 3, border_mode='ref', no_bias=False, layer_name=name + '_conv5_1'))
        self.load_params('vgg19_weights_normalized.h5')

    def get_output(self, input, layer=None):
        input = prep(input)
        if layer is None:
            return super(VGG19, self).get_output(input)
        else:
            return self[:layer](input)

    def load_params(self, param_file=None):
        f = h5py.File(param_file, mode='r')
        trained = [layer[1].value for layer in list(f.items())]
        weight_value_tuples = []
        for p, tp in zip(self.params, trained):
            if len(tp.shape) == 4:
                tp = np.transpose(nn.utils.convert_kernel(tp), (3, 2, 0, 1))
            weight_value_tuples.append((p, tp))
        nn.utils.batch_set_value(weight_value_tuples)
        print('Pretrained weights loaded successfully!')


class Decoder(nn.Sequential):
    def __init__(self, encoder, layer, name='Decoder'):
        super(Decoder, self).__init__(input_shape=encoder[layer-1].output_shape, layer_name=name)
        self.enc = encoder
        self.layer = layer
        dec = nn.Sequential(input_shape=encoder.output_shape, layer_name=name)
        dec.append(
            nn.Conv2DLayer(dec.output_shape, 512, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv1_1'))
        dec.append(nn.UpsamplingLayer(dec.output_shape, 2, method='nearest', layer_name=name+'/up1'))

        dec.append(
            nn.Conv2DLayer(dec.output_shape, 512, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv2_1'))
        dec.append(
            nn.Conv2DLayer(dec.output_shape, 512, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv2_2'))
        dec.append(
            nn.Conv2DLayer(dec.output_shape, 512, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv2_3'))
        dec.append(
            nn.Conv2DLayer(dec.output_shape, 512, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv2_4'))
        dec.append(nn.UpsamplingLayer(dec.output_shape, 2, method='nearest', layer_name=name + '/up2'))

        dec.append(
            nn.Conv2DLayer(dec.output_shape, 256, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv3_1'))
        dec.append(
            nn.Conv2DLayer(dec.output_shape, 256, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv3_2'))
        dec.append(
            nn.Conv2DLayer(dec.output_shape, 256, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv3_3'))
        dec.append(
            nn.Conv2DLayer(dec.output_shape, 256, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv3_4'))
        dec.append(nn.UpsamplingLayer(dec.output_shape, 2, method='nearest', layer_name=name + '/up3'))

        dec.append(
            nn.Conv2DLayer(dec.output_shape, 128, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv4_1'))
        dec.append(
            nn.Conv2DLayer(dec.output_shape, 128, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv4_2'))
        dec.append(nn.UpsamplingLayer(dec.output_shape, 2, method='nearest', layer_name=name + '/up4'))

        dec.append(
            nn.Conv2DLayer(dec.output_shape, 64, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv5_1'))
        dec.append(
            nn.Conv2DLayer(dec.output_shape, 64, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv5_2'))
        dec.append(
            nn.Conv2DLayer(dec.output_shape, 3, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           activation='tanh', layer_name=name + '/output'))
        self.append(dec[len(encoder) - layer:])

    def get_output(self, input, encode=False):
        out = self.enc(input, self.layer) if encode else input
        return super(Decoder, self).get_output(out) / 2. + .5

    def cost(self, input):
        output = self(input, True)
        recon_loss = nn.norm_error(input, output)

        input_ft = self.enc(input, self.layer)
        output_ft = self.enc(output, self.layer)
        feature_loss = nn.norm_error(input_ft, output_ft)
        return recon_loss, feature_loss


class StyleTransfer:
    def __init__(self, encoder, decoders):
        self.enc = encoder
        self.decs = decoders

    def wct(self, input, style):
        svd = T.nlinalg.SVD()

        def _wct(fc, fs):
            mc = T.mean(fc, 1, keepdims=True)
            fc -= mc
            U, S_diag, Vt = svd(T.dot(fc, fc.T))
            S_diag = (S_diag + 1e-8) ** -.5
            fc_hat = T.dot(U * S_diag.dimshuffle('x', 0), T.dot(Vt, fc))

            ms = T.mean(fs, 1, keepdims=True)
            fs -= ms
            U, S_diag, Vt = svd(T.dot(fs, fs.T))
            S_diag = (S_diag + 1e-8) ** .5
            fcs = T.dot(U * S_diag.dimshuffle('x', 0), T.dot(Vt, fc_hat))
            return fcs + ms

        res = theano.scan(_wct, [input.flatten(3), style.flatten(3)])[0]
        res = T.reshape(res, input.shape)
        return res

    def __call__(self, input, style, same_shape=False):
        incomings = input
        for idx, layer in enumerate(indices[::-1]):
            if same_shape:
                inp = T.concatenate((incomings, style))
                out = self.enc(inp, layer)
                x, y = out[:input.shape[0]], out[input.shape[0]:]
            else:
                x, y = self.enc(incomings, layer), self.enc(style, layer)
            fcs = self.wct(x, y)
            incomings = self.decs[-idx-1](fcs)
        return incomings
