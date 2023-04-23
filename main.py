import os
import yaml
import pandas as pd
import torch as tf
import torchaudio as ta
from tqdm import tqdm


CONFIG = None
DEVICE = None


class Config:
    @staticmethod
    def load(path):
        with open(path, 'r') as stream:
            return yaml.safe_load(stream)


class Device:
    @staticmethod
    def choose():
        global CONFIG
        CONFIG_DEVICE = CONFIG['runtime'].get('device', None)

        if CONFIG_DEVICE != 'auto':
            return CONFIG_DEVICE

        return 'cuda' if tf.cuda.is_available() else 'cpu'


class Dataset(tf.utils.data.Dataset):
    @staticmethod
    def load():
        global CONFIG
        CONFIG_TRAIN = CONFIG['dataset']['annotations']['train_csv']
        CONFIG_TEST = CONFIG['dataset']['annotations']['test_csv']

        return Dataset.build(CONFIG_TRAIN), Dataset.build(CONFIG_TEST)

    @staticmethod
    def build(path):
        global CONFIG
        CONFIG_KWARGS = CONFIG['runtime']['loader']

        annotations = pd.read_csv(path)
        dataset = Dataset(annotations)

        return tf.utils.data.DataLoader(dataset, **CONFIG_KWARGS)

    def __init__(self, annotations):
        self.annotations = annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        global CONFIG, DEVICE
        CONFIG_INPUT_COL = CONFIG['dataset']['annotations']['input']
        CONFIG_TARGET_COL = CONFIG['dataset']['annotations']['target']
        CONFIG_FOLDER_PATH = CONFIG['dataset']['folder_path']

        file_path = self.annotations.loc[idx, CONFIG_INPUT_COL]
        full_path = os.path.join(CONFIG_FOLDER_PATH, file_path)
        target = self.annotations.loc[idx, CONFIG_TARGET_COL]

        signal, sr = ta.load(full_path)
        signal.to(DEVICE)

        signal, sr = Dataset.normalize(signal, sr)
        signal = Dataset.transform(signal, sr)

        return signal, target

    @staticmethod
    def normalize(signal, sr):
        global CONFIG
        CONFIG_SAMPLE_RATE = CONFIG['dataset']['normalize']['sample_rate']
        CONFIG_DURATION = CONFIG['dataset']['normalize']['duration']

        duration = CONFIG_SAMPLE_RATE * CONFIG_DURATION
        signal = tf.nn.functional.pad(signal, (0, duration - signal.size()[1]))
        signal = signal.to(DEVICE)

        return signal, sr

    @staticmethod
    def transform(signal, sr):
        global CONFIG, DEVICE
        CONFIG_TRANSOFRM = CONFIG['runtime']['transform']['name']
        CONFIG_KWARGS = CONFIG['runtime']['transform']['params'] or {}

        if CONFIG_TRANSOFRM == 'mel':
            transform = ta.transforms.MelSpectrogram(sample_rate=sr, **CONFIG_KWARGS)
        elif CONFIG_TRANSOFRM == 'mfcc':
            transform = ta.transforms.MFCC(sample_rate=sr, **CONFIG_KWARGS)
        else:
            raise ValueError('Transform not found')

        transform = transform.to(DEVICE)
        signal = transform(signal)

        return signal


class Model:
    @staticmethod
    def use():
        global CONFIG
        CONFIG_MODEL = CONFIG['runtime']['model']['name']

        if CONFIG_MODEL == 'custom':
            model = Model.Model1()
        else:
            raise ValueError('Model not found')

        return model

    @staticmethod
    def loss():
        return tf.nn.CrossEntropyLoss()

    @staticmethod
    def optimizer(model):
        global CONFIG
        CONFIG_LEARNING_RATE = CONFIG['runtime']['learning_rate']
        return tf.optim.Adam(model.parameters(), lr=CONFIG_LEARNING_RATE)

    class Model1(tf.nn.Module):
        def __init__(self):
            super().__init__()

            self.conv1 = tf.nn.Sequential(
                tf.nn.Conv2d(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=2
                ),
                tf.nn.ReLU(),
                tf.nn.MaxPool2d(kernel_size=2)
            )

            self.conv2 = tf.nn.Sequential(
                tf.nn.Conv2d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=2
                ),
                tf.nn.ReLU(),
                tf.nn.MaxPool2d(kernel_size=2)
            )

            self.conv3 = tf.nn.Sequential(
                tf.nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=2
                ),
                tf.nn.ReLU(),
                tf.nn.MaxPool2d(kernel_size=2)
            )

            self.conv4 = tf.nn.Sequential(
                tf.nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=2
                ),
                tf.nn.ReLU(),
                tf.nn.MaxPool2d(kernel_size=2)
            )

            self.flatten = tf.nn.Flatten()
            self.linear = tf.nn.Linear(2304, 10)
            self.softmax = tf.nn.Softmax(dim=1)

        def forward(self, data):
            x = self.conv1(data)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.flatten(x)
            logits = self.linear(x)
            predictions = self.softmax(logits)
            return predictions

    class Model2(tf.nn.Module):
        pass

    class Model3(tf.nn.Module):
        pass


class AudioCNNClassifier:
    @staticmethod
    def main():
        global CONFIG, DEVICE
        CONFIG = Config.load('./config.yml')
        DEVICE = Device.choose()

        print(f'Device: {DEVICE.upper()}')

        model_obj = Model.use().to(DEVICE)
        loss_fn = Model.loss()
        optimiser = Model.optimizer(model_obj)

        train_batches, test_batches = Dataset.load()

        AudioCNNClassifier.train_all_epochs(model_obj, loss_fn, optimiser, train_batches)

        tf.save(model_obj.state_dict(), 'model/audio.pth')

    @staticmethod
    def train_all_epochs(model_obj, loss_fn, optimiser, train_batches):
        global CONFIG
        CONFIG_EPOCHS = CONFIG['runtime']['epochs']
        progress = tqdm(range(CONFIG_EPOCHS))

        for i in progress:
            loss = AudioCNNClassifier.train_single_epoch(model_obj, loss_fn, optimiser, train_batches)
            progress.set_description(f'Epoch: {i} | Loss: {loss}')

        print('Finished training')

    @staticmethod
    def train_single_epoch(model_obj, loss_fn, optimiser, train_batches):
        global DEVICE
        for x, y in train_batches:
            x, y = x.to(DEVICE), y.to(DEVICE)

            prediction = model_obj(x)
            loss_val = loss_fn(prediction, y)

            optimiser.zero_grad()
            loss_val.backward()
            optimiser.step()

        return loss_val.item()


if __name__ == '__main__':
    AudioCNNClassifier.main()
