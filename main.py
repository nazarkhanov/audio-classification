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

        tf.save(model_obj.state_dict(), 'model/audio.pth')

    @staticmethod
    def _train_all_epochs(model_obj, loss_fn, optimiser, train_batches):
        CONFIG_EPOCHS = g.CONFIG['runtime']['epochs']
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
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='train | test')
    args = parser.parse_args()

    if args.action == 'train':
        Main.train()
    elif args.action == 'test':
        Main.test()
