import os
import torch as tf
import torch.nn as nn
import torchaudio as ta
import pandas as pd
import tools.globals as g


class Dataset(tf.utils.data.Dataset):
    @staticmethod
    def read_annotations():
        CONFIG_TRAIN = g.CONFIG['dataset']['annotations']['train_csv']
        CONFIG_TEST = g.CONFIG['dataset']['annotations']['test_csv']

        train_annotations = pd.read_csv(CONFIG_TRAIN)
        test_annotations = pd.read_csv(CONFIG_TEST)

        return train_annotations, test_annotations

    @staticmethod
    def load_batches(annotations):
        CONFIG_KWARGS = g.CONFIG['runtime']['loader']

        train_annotations, test_annotations = annotations

        train_dataset = Dataset(train_annotations)
        train_batches = tf.utils.data.DataLoader(train_dataset, **CONFIG_KWARGS)

        test_dataset = Dataset(train_annotations)
        test_batches = tf.utils.data.DataLoader(test_dataset, **CONFIG_KWARGS)

        return train_batches, test_batches

    @staticmethod
    def count_classes(annotations):
        CONFIG_TARGET_COL = g.CONFIG['dataset']['annotations']['target']

        train_annotations, test_annotations = annotations
        df = pd.concat([train_annotations, test_annotations]).nunique()

        return df[CONFIG_TARGET_COL]

    def __init__(self, annotations):
        self.annotations = annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        CONFIG_INPUT_COL = g.CONFIG['dataset']['annotations']['input']
        CONFIG_TARGET_COL = g.CONFIG['dataset']['annotations']['target']
        CONFIG_FOLDER_PATH = g.CONFIG['dataset']['folder_path']

        file_path = self.annotations.loc[idx, CONFIG_INPUT_COL]
        full_path = os.path.join(CONFIG_FOLDER_PATH, file_path)
        target = self.annotations.loc[idx, CONFIG_TARGET_COL]

        signal, sr = ta.load(full_path)
        signal.to(g.DEVICE)

        signal, sr = Dataset.normalize(signal, sr)
        signal = Dataset.transform(signal, sr)

        return signal, target

    @staticmethod
    def normalize(signal, sr):
        CONFIG_SAMPLE_RATE = g.CONFIG['dataset']['normalize']['sample_rate']
        CONFIG_DURATION = g.CONFIG['dataset']['normalize']['duration']

        duration = CONFIG_SAMPLE_RATE * CONFIG_DURATION
        signal = nn.functional.pad(signal, (0, duration - signal.size()[1]))
        signal = signal.to(g.DEVICE)

        return signal, sr

    @staticmethod
    def transform(signal, sr):
        CONFIG_TRANSFORM = g.CONFIG['runtime']['transform']['name']
        CONFIG_KWARGS = g.CONFIG['runtime']['transform']['params'] or {}

        if CONFIG_TRANSFORM == 'mel':
            if (g.TRANSFORM is None) or (type(g.TRANSFORM) is not ta.transforms.MelSpectrogram):
                g.TRANSFORM = ta.transforms.MelSpectrogram(sample_rate=sr, **CONFIG_KWARGS).to(g.DEVICE)
        elif CONFIG_TRANSFORM == 'mfcc':
            if (g.TRANSFORM is None) or (type(g.TRANSFORM) is not ta.transforms.MFCC):
                g.TRANSFORM = ta.transforms.MFCC(sample_rate=sr, **CONFIG_KWARGS).to(g.DEVICE)
        else:
            raise ValueError('Transform not found')

        transform = g.TRANSFORM
        signal = transform(signal)

        return signal
