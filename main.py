import pathlib
import argparse
import torch as tf
import tools.globals as g
from tqdm import tqdm
from tools.config import Config
from tools.device import Device
from tools.dataset import Dataset
from models import Model


class Main:
    @staticmethod
    def train():
        g.CONFIG = Config.load('./config.yml')
        g.DEVICE = Device.choose()

        print(f'Device: {g.DEVICE.upper()}')

        annotations = Dataset.read_annotations()
        train_batches, test_batches = Dataset.load_batches(annotations)
        num_classes = Dataset.count_classes(annotations)

        model_obj = Model.init(num_classes).to(g.DEVICE)
        loss_fn = Model.loss()
        optimiser = Model.optimizer(model_obj)

        Main._train_all_epochs(model_obj, loss_fn, optimiser, train_batches)

        path = g.CONFIG['model']['folder_path']
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        path = str(pathlib.PurePath(path, 'audio.pth'))
        tf.save(model_obj.state_dict(), path)

    @staticmethod
    def _train_all_epochs(model_obj, loss_fn, optimiser, train_batches):
        CONFIG_EPOCHS = g.CONFIG['model']['trainer']['epochs']
        progress = tqdm(range(CONFIG_EPOCHS))

        for i in progress:
            loss = Main._train_single_epoch(model_obj, loss_fn, optimiser, train_batches)
            progress.set_description(f'Epoch: {i} | Loss: {loss}')

        print('Finished training')

    @staticmethod
    def _train_single_epoch(model_obj, loss_fn, optimiser, train_batches):
        for x, y in train_batches:
            x, y = x.to(g.DEVICE), y.to(g.DEVICE)

            prediction = model_obj(x)
            loss_val = loss_fn(prediction, y)

            optimiser.zero_grad()
            loss_val.backward()
            optimiser.step()

        return loss_val.item()

    @staticmethod
    def test():
        g.CONFIG = Config.load('./config.yml')
        g.DEVICE = Device.choose()

        print(f'Device: {g.DEVICE.upper()}')

        annotations = Dataset.read_annotations()
        train_batches, test_batches = Dataset.load_batches(annotations)
        num_classes = Dataset.count_classes(annotations)

        model_obj = Model.init(num_classes).to(g.DEVICE)

        path = g.CONFIG['model']['folder_path']
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        path = str(pathlib.PurePath(path, 'audio.pth'))
        model_obj.load_state_dict(tf.load(path))

        model_obj.eval()

        total = 0
        right = 0

        with tf.no_grad():
            with tqdm(len(test_batches)) as progress:
                for x, y in test_batches:
                    x, y = x.to(g.DEVICE), y.to(g.DEVICE)

                    predictions = model_obj(x)
                    predictions = predictions.argmax(dim=1)

                    total += int(y.size()[0])
                    right += Main._count_right_predictions(predictions, y)

                    progress.update(1)
                    progress.set_description(f'Right predicted: {right} | Total count:{total} | Accuracy: {right / total}')

    @staticmethod
    def _count_right_predictions(predicted, expected):
        return int(tf.eq(predicted, expected).sum())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='train | test')
    args = parser.parse_args()

    if args.action == 'train':
        Main.train()
    elif args.action == 'test':
        Main.test()
