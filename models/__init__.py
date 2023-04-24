import torch as tf
import torch.nn as nn
import tools.globals as g
from models.vgg import VGG
from models.net import Net
from models.alexnet import AlexNet


class Model:
    @staticmethod
    def init(num_classes):
        CONFIG_MODEL = g.CONFIG['runtime']['model']['name']

        if CONFIG_MODEL == 'vgg':
            model = VGG(num_classes)
        elif CONFIG_MODEL == 'net':
            model = Net(num_classes)
        elif CONFIG_MODEL == 'alexnet':
            model = AlexNet(num_classes)
        else:
            raise ValueError('Model not found')

        return model

    @staticmethod
    def loss():
        return nn.CrossEntropyLoss()

    @staticmethod
    def optimizer(model):
        CONFIG_LEARNING_RATE = g.CONFIG['runtime']['learning_rate']
        return tf.optim.Adam(model.parameters(), lr=CONFIG_LEARNING_RATE)
