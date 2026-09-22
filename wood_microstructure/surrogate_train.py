"""
Training of the surrogate model for the undistorted to distorted microstructure mapping.
"""
import copy
import json
import os
import sys

import numpy as np

from . import myio
from .clocks import Clock
from .params import TrainParams
from .pipeline import Pipeline

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print('PyTorch is not installed, required for surrogate model training.')
    sys.exit(1)

try:
    from torchvision.transforms import ToPILImage, ToTensor
except ImportError:
    print('torchvision is not installed, required for surrogate model training.')
    sys.exit(1)


class ImageDataset(Dataset):
    def __init__(self, dir_nodist, dir_u_map, dir_v_map, dir_dist, padded):
        # TODO:
        # - should this be harcoded?
        # - Are all images expected to have the same size?
        # - Or be bigger/smaller than this size?
        self.rows = 900
        self.columns = 1600
        # self.rows: int = None
        # self.columns: int = None

        self.padded = padded

        self.data_dir_nodist = dir_nodist
        self.data_dir_u_map = dir_u_map
        self.data_dir_v_map = dir_v_map
        self.data_dir_dist = dir_dist

        self.images_nodist = sorted(self.getfiles(dir_nodist,True))
        self.u_maps = sorted(self.getfiles(dir_u_map,True))
        self.v_maps = sorted(self.getfiles(dir_v_map,True))
        if dir_dist:
            self.images_dist = sorted(self.getfiles(dir_dist,True))
        self.files = sorted(self.getfiles(dir_nodist,False))

    def __len__(self):
        return len(self.images_nodist)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, str]:
        image_nodist = myio.read_slice(self.images_nodist[index])
        # if self.rows is None or self.columns is None:
        #     self.rows, self.columns = image_nodist.size[1], image_nodist.size[0]
        # else:
        #     rows, columns = image_nodist.size[1], image_nodist.size[0]
        #     if rows != self.rows or columns != self.columns:
        #         raise ValueError(
        #             f'Image size mismatch at idx {index}: '
        #             f'expected ({self.rows}, {self.columns}), got ({rows}, {columns})'
        #         )
        image_nodist = ToTensor()(image_nodist)
        # if self.padded == True:
        #     image_nodist = self.zeropadding(image_nodist)

        # u_map = pd.read_csv(self.u_maps[index], header=None)
        u_map = myio.read_slice(self.u_maps[index])
        u_map = torch.tensor(u_map.astype(np.float32))
        u_map = torch.unsqueeze(u_map,0)
        # if self.padded == True:
        #     u_map = self.zeropadding(u_map)

        # v_map = pd.read_csv(self.v_maps[index], header=None)
        v_map = myio.read_slice(self.v_maps[index])
        v_map = torch.tensor(v_map.astype(np.float32))
        v_map = torch.unsqueeze(v_map,0)
        # if self.padded == True:
        #     v_map = self.zeropadding(v_map)

        if self.data_dir_dist:
            image_dist = myio.read_slice(self.images_dist[index])
            image_dist = ToTensor()(image_dist)
            # if self.padded == True:
            #     image_dist = self.zeropadding(image_dist)
        else:
            image_dist = None

        return image_nodist, u_map, v_map, image_dist, self.files[index]

    # def zeropadding(self, unpadded):
    #     lsize = unpadded.size()
    #     sh = lsize[1]
    #     sw = lsize[2]
    #     zeroPad = nn.ZeroPad2d((self.columns-sw,0,self.rows-sh,0))
    #     padded = zeroPad(unpadded)
    #     return padded

    def getfiles(self, path, is_top):
        list = []
        if is_top == True:
            for (root, dirs, files) in os.walk(path):
                for file in files:
                    list.append(os.path.join(root,file))
        else:
            for (root, dirs, files) in os.walk(path):
                for file in files:
                    list.append(os.path.join(root[len(path)+1:],file))
        return list


class TrainSurrogate(Pipeline[TrainParams]):
    ParamsClass = TrainParams
    save_prefix = 'train_surrogate'
    logname = 'train'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.init_torch()

        self.model: 'torch.nn.Module' | None = None
        self.epochs = []
        self.loss = []

        self.epoch: int = 0

        self.loss_curve_file = os.path.join(self.root_dir, 'loss_curve.dat')

    def init_torch(self):
        """Initialize PyTorch and check for GPU availability."""
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.logger.info('PyTorch initialized successfully. Using device: %s', self.device)
        if self.device == 'cpu':
            self.logger.warning('GPU not available. Training will be slower on CPU.')

    def init_pipeline(self):
        """Initialize pipeline"""
        self.tasks = tasks = []

        tasks.append((self.init_model, [], {}, True))
        tasks.append((self.init_dataset, [], {}, True))
        tasks.append((self.run_training, [], {}, True))
        tasks.append((self.save_best_model, [], {}, True))
        tasks.append((self.test_loss, [], {}, True))

    def init_model(self):
        """Initialize the surrogate model"""
        from .surrogate import U_Net

        self.model = U_Net()
        self.model.to(self.device)

        if self.params.pretrain_weights:
            weights = torch.load(self.params.pretrain_weights, map_location=self.device)
            self.model.load_state_dict(weights)

        if self.params.frozen_layers:
            frozen_layers = set(self.params.frozen_layers)
            used = set()
            for name, param in self.model.named_parameters():
                for prefix in frozen_layers:
                    if name.startswith(prefix):
                        param.requires_grad = False
                        used.add(prefix)
                        self.logger.info('Freezing layer: %s', name)
                        break
            if unused := frozen_layers - used:
                self.logger.warning('Unused frozen layers: %s', ', '.join(unused))

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate)

        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')
        self.best_model_weights = None

        with open(self.loss_curve_file, 'w') as f:
            f.write('#epoch training_loss validation_loss\n')

    @staticmethod
    def _custom_collate(
        data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, str]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, list[str]]:
        """Custom collate function to handle variable image sizes in the dataset."""
        max_h = 0
        max_w = 0
        for item in data:
            if item[0].size()[1] > max_h:
                max_h = item[0].size()[1]
            if item[0].size()[2] > max_w:
                max_w = item[0].size()[2]

        X1 = torch.zeros(len(data),1,max_h,max_w)
        X2 = torch.zeros(len(data),1,max_h,max_w)
        X3 = torch.zeros(len(data),1,max_h,max_w)
        y = torch.zeros(len(data),1,max_h,max_w)
        file_name = []

        for i, item in enumerate(data):
            sh = item[0].size()[1]
            sw = item[0].size()[2]
            X1[i] = torch.cat(
                [
                    torch.cat([item[0], torch.zeros(1,max_h-sh,sw)], axis=1),
                    torch.zeros(1,max_h,max_w-sw)
                ],
                axis=2
            )
            X2[i] = torch.cat([
                    torch.cat([item[1], torch.zeros(1,max_h-sh,sw)], axis=1),
                    torch.zeros(1,max_h,max_w-sw)
                ],
                axis=2
            )
            X3[i] = torch.cat([
                    torch.cat([item[2], torch.zeros(1,max_h-sh,sw)], axis=1),
                    torch.zeros(1,max_h,max_w-sw)
                ],
                axis=2
            )
            y[i] = torch.cat([
                    torch.cat([item[3], torch.zeros(1,max_h-sh,sw)], axis=1),
                    torch.zeros(1,max_h,max_w-sw)
                ],
                axis=2
            )
            file_name.append(item[4])

        return X1, X2, X3, y, file_name

    def init_dataset(self):
        """Initialize the dataset for training"""
        params = self.params

        paths = {}
        for dname, root in [
                ('train', params.train_dir),
                ('validation', params.validation_dir),
                ('test', params.test_dir),
            ]:
            nodist = os.path.join(root, params.backbone_subdir)
            dist = os.path.join(root, params.distorted_subdir)
            u_map = os.path.join(root, params.u_map_subdir)
            v_map = os.path.join(root, params.v_map_subdir)
            paths[dname] = [nodist, u_map, v_map, dist]

        self.train_set = ImageDataset(*paths['train'], padded=False)
        self.valid_set = ImageDataset(*paths['validation'], padded=False)
        self.test_set = ImageDataset(*paths['test'], padded=False)

        train_size = len(self.train_set)
        valid_size = len(self.valid_set)
        test_size = len(self.test_set)
        self.logger.info('Training set size: %d samples', train_size)
        self.logger.info('Validation set size: %d samples', valid_size)
        self.logger.info('Test set size: %d samples', test_size)

        if train_size == 0:
            self.logger.error('Training set is empty. Please check the training data directory.')
            sys.exit(1)
        if valid_size == 0:
            self.logger.error('Validation set is empty. Please check the validation data directory.')
            sys.exit(1)
        if test_size == 0:
            self.logger.warning('Test set is empty. Will skip final test. Please check the test data directory.')

        self.train_loader = DataLoader(
            self.train_set, batch_size=self.params.training_batch_size,
            shuffle=True, collate_fn=self._custom_collate, num_workers=params.training_workers
        )
        self.valid_loader = DataLoader(
            self.valid_set, batch_size=self.params.validation_batch_size,
            shuffle=False, collate_fn=self._custom_collate, num_workers=params.validation_workers
        )

    @Clock.register(['Training'])
    def train_one_epoch(self) -> float:
        """Train the model for one epoch and return the average training loss"""
        # self.model.train()
        # total_loss = 0.0
        # for batch_idx, (X1, X2, X3, y, _) in enumerate(self.train_loader):
        #     X1, X2, X3, y = X1.to(self.device), X2.to(self.device), X3.to(self.device), y.to(self.device)

        #     self.optimizer.zero_grad()
        #     outputs = self.model(X1, X2, X3)
        #     loss = self.loss_fn(outputs, y)
        #     loss.backward()
        #     self.optimizer.step()

        #     total_loss += loss.item()

        # avg_loss = total_loss / len(self.train_loader)
        # self.logger.info('Epoch %d: Training loss: %.6f', self.epoch + 1, avg_loss)
        # return avg_loss

        dataloader = self.train_loader
        # train_batch_size = self.params.training_batch_size
        device = self.device

        # size = len(dataloader.dataset)
        num_batches = len(dataloader)
        total_loss = 0.0
        for batch, (X1, X2, X3, y, file_name) in enumerate(dataloader):
            X1 = X1.to(device)
            X2 = X2.to(device)
            X3 = X3.to(device)
            y = y.to(device)

            self.optimizer.zero_grad()
            pred = self.model(X1, X2, X3)
            loss = self.loss_fn(pred, y)

            loss.backward()
            self.optimizer.step()

            loss = loss.item()
            total_loss += loss
            self.logger.debug('Batch %d/%d: loss: %.6f', batch + 1, num_batches, loss)

        avg_loss = total_loss / num_batches
        self.logger.info('Epoch %d: Training loss: %.6f', self.epoch + 1, avg_loss)
        return avg_loss

    @Clock.register(['Validation'])
    def validate_one_epoch(self) -> float:
        """Validate the model for one epoch and return the average validation loss"""
        dataloader = self.valid_loader
        device = self.device

        num_batches = len(dataloader)
        val_loss = 0.0
        with torch.no_grad():
            for (X1, X2, X3, y, file_name) in dataloader:
                X1 = X1.to(device)
                X2 = X2.to(device)
                X3 = X3.to(device)
                y = y.to(device)

                pred = self.model(X1,X2,X3)
                val_loss += self.loss_fn(pred, y).item()
        val_loss /= num_batches
        self.logger.info('Epoch %d: Validation loss: %.6f', self.epoch + 1, val_loss)
        return val_loss

    def run_training(self):
        """Run the training loop for the surrogate model"""
        patience = 0
        for epoch in self.track_step(range(self.params.epochs), description='Training epochs'):
            self.epoch = epoch
            self.logger.info('Starting epoch %d/%d', epoch + 1, self.params.epochs)
            train_loss = self.train_one_epoch()
            val_loss = self.validate_one_epoch()

            self.epochs.append(epoch)
            self.loss.append((train_loss, val_loss))

            with open(self.loss_curve_file, 'a') as f:
                f.write(f'{epoch:>6d} {train_loss:>11.7f} {val_loss:>11.7f}\n')

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_train_loss = train_loss
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                # best_model_file = os.path.join(self.root_dir, 'best_model.pth')
                # torch.save(self.best_model_weights, best_model_file)
                patience = 0
                self.logger.info('Best model saved at epoch %d with validation loss %.6f', epoch + 1, val_loss)
            else:
                patience += 1
                self.logger.debug('No improvement in validation loss. Patience: %d/%d', patience, self.params.patience)
                if patience >= self.params.patience:
                    self.logger.info('Early stopping at epoch %d', epoch + 1)
                    break

            if self.params.save_interval > 0 and (epoch + 1) % self.params.save_interval == 0:
                self.save_best_model()
                self._save_model_weights(os.path.join(self.root_dir, f'model_epoch_{epoch + 1}.pth'))

    @Clock.register(['I/O', 'Model Saving'])
    def _save_model_weights(self, filename: str):
        """Save the model weights to a file"""
        torch.save(self.model.state_dict(), filename)
        self.logger.info('Model weights saved to %s', filename)

    def save_best_model(self):
        """Save the trained surrogate model"""
        self.model.load_state_dict(self.best_model_weights)
        self._save_model_weights(os.path.join(self.root_dir, 'best_model.pth'))

        torch.save(self.model.state_dict(), os.path.join(self.root_dir, 'best_model.pth'))

    @Clock.register(['Testing'])
    def test_loss(self):
        """Evaluate the model on a test set and return the loss"""
        dataset = self.test_set
        device = self.device

        if dataset is None or len(dataset) == 0:
            self.logger.warning('Test set is empty. Skipping test loss evaluation.')
            return

        transform = ToPILImage()
        num_samples = len(dataset)
        test_loss = 0.0

        test_outdir = os.path.join(self.root_dir, 'test_output')

        for i, sample in enumerate(dataset):
            pred = self.model(torch.unsqueeze(sample[0],0).to(device),torch.unsqueeze(sample[1],0).to(device),torch.unsqueeze(sample[2],0).to(device))
            test_loss += self.loss_fn(pred, torch.unsqueeze(sample[3], 0).to(device)).item()

            img = transform(torch.squeeze(pred))
            file_out = os.path.join(test_outdir, str(sample[4]))
            img.save(file_out)
            self.logger.debug('Saved %s test output image: %s', i, file_out)

        test_loss /= num_samples

        losses = {
            'train_loss': self.best_train_loss,
            'validation_loss': self.best_val_loss,
            'test_loss': test_loss,
        }

        losses_file = os.path.join(self.root_dir, 'losses.json')
        with open(losses_file, 'w') as f:
            json.dump(losses, f, indent=4)
