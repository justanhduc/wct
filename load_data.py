import os
import neuralnet as nn
import numpy as np
from scipy import misc
from random import shuffle


def prep_image(im, size=256, resize=512):
    h, w, _ = im.shape
    if h < w:
        new_sh = (resize, int(w * resize / h))
    else:
        new_sh = (int(h * resize / w), resize)
    im = misc.imresize(im, new_sh, interp='bicubic')

    # Crop random
    im = nn.utils.crop_random(im, size)
    im = im.astype('float32')
    return np.transpose(im[None], (0, 3, 1, 2)) / 255.


def prep_image_test(im):
    im = im.astype('float32')
    return np.transpose(im[None], (0, 3, 1, 2)) / 255.


class DataManagerStyleTransfer(nn.DataManager):
    def __init__(self, placeholders, path, bs, n_epochs, shuffle=False, **kwargs):
        super(DataManagerStyleTransfer, self).__init__(None, placeholders, path, bs, n_epochs, shuffle=shuffle, **kwargs)
        self.load_data()
        self.num_val_imgs = kwargs.get('num_val_imgs')
        self.input_shape = kwargs.get('input_shape')

    def load_data(self):
        source = os.listdir(self.path[0])
        source = [self.path[0] + '/' + f for f in source]
        if 'val' in self.path[0] or 'test' in self.path[0]:
            source = source[:self.num_val_imgs]

        style = os.listdir(self.path[1])
        style = [self.path[1] + '/' + f for f in style]

        shuffle(source)
        shuffle(style)

        self.dataset = (source, style)
        self.data_size = min(len(source), len(style))

    def generator(self):
        source, style = list(self.dataset[0]), list(self.dataset[1])

        if self.shuffle:
            shuffle(source)
            shuffle(style)

        if len(source) > len(style):
            source = source[:len(style)]
        else:
            style = style[:len(source)]

        for idx in range(0, self.data_size, self.batch_size):
            source_batch = source[idx:idx + self.batch_size]
            style_batch = style[idx:idx + self.batch_size]
            imgs, stys = [], []
            exit = False
            for sou_f, sty_f in zip(source_batch, style_batch):
                try:
                    img, sty = misc.imread(sou_f), misc.imread(sty_f)
                except FileNotFoundError:
                    exit = True
                    break

                if len(img.shape) < 3 or len(sty.shape) < 3:
                    exit = True
                    break

                imgs.append(prep_image(img, self.input_shape[-1], self.input_shape[-1] * 2))
                stys.append(prep_image(sty, self.input_shape[-1], self.input_shape[-1] * 2))

            if exit:
                continue

            imgs = np.concatenate(imgs)
            stys = np.concatenate(stys)
            yield imgs, stys


class DataManager(nn.DataManager):
    def __init__(self, placeholders, path, bs, n_epochs, shuffle=False, **kwargs):
        super(DataManager, self).__init__(None, placeholders, path, bs, n_epochs, shuffle=shuffle, **kwargs)
        self.load_data()
        self.input_shape = kwargs.get('input_shape')

    def load_data(self):
        source = os.listdir(self.path)
        source = [self.path + '/' + f for f in source]
        shuffle(source)

        self.dataset = source
        self.data_size = len(source)

    def generator(self):
        source = list(self.dataset)

        if self.shuffle:
            shuffle(source)

        for idx in range(0, self.data_size, self.batch_size):
            source_batch = source[idx:idx + self.batch_size]
            imgs, stys = [], []
            exit = False
            for sou_f in source_batch:
                try:
                    img = misc.imread(sou_f)
                except FileNotFoundError:
                    exit = True
                    break

                if len(img.shape) < 3:
                    exit = True
                    break

                img = prep_image(img, self.input_shape[-1], self.input_shape[-1] * 2)
                if img.shape != (1,) + self.input_shape[1:]:
                    exit = True
                    break

                imgs.append(img)

            if exit:
                continue

            imgs = np.concatenate(imgs)

            if imgs.shape != (self.batch_size,) + input_shape[1:]:
                continue

            yield imgs
