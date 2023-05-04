import torch as tf
import tools.globals as g


class Device:
    @staticmethod
    def choose():
        CONFIG_DEVICE = g.CONFIG['model'].get('device', None)

        if CONFIG_DEVICE != 'auto':
            return CONFIG_DEVICE

        return 'cuda' if tf.cuda.is_available() else 'cpu'
