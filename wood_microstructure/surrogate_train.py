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
    from torch.utils.data import DataLoader, Dataset, Subset, random_split
except ImportError:
    print('PyTorch is not installed, required for surrogate model training.')
    sys.exit(1)

try:
    from torchvision.transforms import ToPILImage, ToTensor
except ImportError:
    print('torchvision is not installed, required for surrogate model training.')
    sys.exit(1)


class ImageDataset(Dataset):
    def __init__(self, dir_nodist, dir_u_map, dir_v_map, dir_dist):
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
        image_nodist = ToTensor()(image_nodist)

        u_map = myio.read_slice(self.u_maps[index])
        u_map = torch.tensor(u_map.astype(np.float32))
        u_map = torch.unsqueeze(u_map,0)

        v_map = myio.read_slice(self.v_maps[index])
        v_map = torch.tensor(v_map.astype(np.float32))
        v_map = torch.unsqueeze(v_map,0)

        if self.data_dir_dist:
            image_dist = myio.read_slice(self.images_dist[index])
            image_dist = ToTensor()(image_dist)
        else:
            image_dist = None

        return image_nodist, u_map, v_map, image_dist, self.files[index]

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

        self.cross_validation = self.params.cross_validation
        self.num_folds = self.params.num_folds
        self.train_ratio = self.params.train_ratio
        self.valid_ratio = 1 - self.train_ratio

        if self.cross_validation and self.num_folds < 2:
            self.logger.warning(
                'Cross-validation is enabled but num_folds is less than 2. Setting num_folds to 5 for cross-validation.'
            )
            self.num_folds = 5

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
        tasks.append((self.save_losses, [], {}, True))

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
        if not os.path.exists(params.train_dir):
            self.logger.error('Training data directory does not exist: %s', params.train_dir)
            sys.exit(1)

        for dname, root in [
                ('train', params.train_dir),
                ('validation', params.validation_dir),
                ('test', params.test_dir),
            ]:
            if not root:
                continue
            nodist = os.path.join(root, params.backbone_subdir)
            dist = os.path.join(root, params.distorted_subdir)
            u_map = os.path.join(root, params.u_map_subdir)
            v_map = os.path.join(root, params.v_map_subdir)
            paths[dname] = [nodist, u_map, v_map, dist]

        num_folds = 1
        self.train_set = ImageDataset(*paths['train'])
        if 'validation' not in paths or 'test' not in paths or self.cross_validation:
            if not self.cross_validation:
                self.logger.warning(
                    'Validation or test data directory not provided. FORCING cross-validation split from training data.'
                )
            else:
                self.logger.info('Cross-validation split enabled. Splitting training data into folds.')
            if 'validation' in paths:
                self.logger.warning(
                    'Validation data is provided but will be ignored due to cross-validation split.'
                )
            if 'test' in paths:
                self.logger.warning(
                    'Test data is provided but will be ignored due to cross-validation split.'
                )

            try:
                from sklearn.model_selection import KFold
            except ImportError:
                self.logger.error('scikit-learn is not installed, required for cross-validation split.')
                sys.exit(1)

            self.logger.info(
                (
                    'Performing K-Fold cross-validation with %d folds. '
                    'Training data will be split into %.2f%% training and %.2f%% validation for each fold.'
                ),
                self.num_folds,
                self.train_ratio * 100,
                self.valid_ratio * 100
            )

            self.cross_validation = True
            num_folds = self.num_folds

            kfold = KFold(n_splits = num_folds, shuffle=True, random_state=1)
            self.kfold_split = list(kfold.split(self.train_set))
        else:
            self.valid_set = ImageDataset(*paths['validation'])
            self.test_set = ImageDataset(*paths['test'])
            self.kfold_split = [(0, 0)]  # Dummy split for consistency

        self.logger.info('Initialized dataset with %d folds', num_folds)
        self.num_folds = num_folds
        self.train_losses = np.zeros(num_folds)
        self.valid_losses = np.zeros(num_folds)
        self.test_losses = np.zeros(num_folds)

    @Clock.register(['Training'])
    def train_one_epoch(self) -> float:
        """Train the model for one epoch and return the average training loss"""
        dataloader = self.train_loader
        device = self.device

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
        return val_loss

    def run_training(self):
        """Run the training loop for the surrogate model"""
        initial_weights = copy.deepcopy(self.model.state_dict())
        for fold, (train_index, test_index) in self.track_step(
                enumerate(self.kfold_split),
                description='Cross-validation folds',
                total=self.num_folds
            ):
            if self.cross_validation:
                subset = Subset(self.train_set, train_index)
                train_subset, valid_subset = random_split(subset, [self.train_ratio, self.valid_ratio])
                test_subset = Subset(self.train_set, test_index)
                self.train_loader = DataLoader(
                    train_subset, batch_size=self.params.training_batch_size,
                    shuffle=True, collate_fn=self._custom_collate, num_workers=self.params.training_workers
                )
                self.valid_loader = DataLoader(
                    valid_subset, batch_size=self.params.validation_batch_size,
                    shuffle=False, collate_fn=self._custom_collate, num_workers=self.params.validation_workers
                )
            else:
                self.train_loader = DataLoader(
                    self.train_set, batch_size=self.params.training_batch_size,
                    shuffle=True, collate_fn=self._custom_collate, num_workers=self.params.training_workers
                )
                self.valid_loader = DataLoader(
                    self.valid_set, batch_size=self.params.validation_batch_size,
                    shuffle=False, collate_fn=self._custom_collate, num_workers=self.params.validation_workers
                )
                test_subset = self.test_set

            best_train_loss = float('inf')
            best_val_loss = float('inf')
            best_model_weights = None

            train_size = len(self.train_loader.dataset)
            valid_size = len(self.valid_loader.dataset)
            test_size = len(test_subset) if test_subset is not None else 0
            if train_size == 0:
                self.logger.error(f'Fold {fold + 1}: Training set is empty. Please check the training data directory.')
                continue  # Skip this fold instead of exiting, to allow other folds to run
            if valid_size == 0:
                self.logger.error(f'Fold {fold + 1}: Validation set is empty. Please check the validation data directory.')
                continue  # Skip this fold instead of exiting, to allow other folds to run
            if test_size == 0:
                self.logger.warning(f'Fold {fold + 1}: Test set is empty. Will skip final test. Please check the test data directory.')

            self.logger.info(
                'Starting fold %d/%d: Train/Valid/Test samples: %d/%d/%d',
                fold + 1, self.num_folds, train_size, valid_size, test_size
            )

            patience = 0
            loss_curve_file = os.path.join(self.root_dir, f'loss_curve_fold_{fold + 1}.dat')
            with open(loss_curve_file, 'w') as f:
                f.write('#epoch training_loss validation_loss\n')

            for epoch in self.track_step(range(self.params.epochs), description='Training epochs'):
                self.logger.info('Fold %d: Starting epoch %d/%d', fold + 1, epoch + 1, self.params.epochs)
                train_loss = self.train_one_epoch()
                val_loss = self.validate_one_epoch()

                self.logger.info(
                    'Fold %d: Epoch %d: Training loss: %.6f, Validation loss: %.6f',
                    fold + 1, epoch + 1, train_loss, val_loss
                )

                with open(loss_curve_file, 'a') as f:
                    f.write(f'{epoch:>6d} {train_loss:>11.7f} {val_loss:>11.7f}\n')

                if val_loss < best_val_loss:
                    patience = 0
                    best_val_loss = val_loss
                    best_train_loss = train_loss
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                else:
                    patience += 1
                    self.logger.debug(
                        'No improvement in validation loss. Patience: %d/%d', patience, self.params.patience
                    )
                    if patience >= self.params.patience:
                        self.logger.info('Early stopping at epoch %d', epoch + 1)
                        break

                if self.params.save_interval > 0 and (epoch + 1) % self.params.save_interval == 0:
                    self._save_model_weights(os.path.join(self.root_dir, f'model_epoch_{epoch + 1}.pth'))

            self.model.load_state_dict(best_model_weights)
            test_loss = self._test_loss(test_subset, fold)

            self._save_model_weights(os.path.join(self.root_dir, f'best_model_fold_{fold + 1}.pth'))

            self.train_losses[fold] = best_train_loss
            self.valid_losses[fold] = best_val_loss
            self.test_losses[fold] = test_loss

            if best_val_loss < self.best_val_loss:
                self.best_val_loss = best_val_loss
                self.best_train_loss = best_train_loss
                self.best_model_weights = copy.deepcopy(best_model_weights)

            self.model.load_state_dict(initial_weights)

    @Clock.register(['I/O', 'Model Saving'])
    def _save_model_weights(self, filename: str):
        """Save the model weights to a file"""
        torch.save(self.model.state_dict(), filename)
        self.logger.info('Model weights saved to %s', filename)

    def save_best_model(self):
        """Save the trained surrogate model"""
        self.model.load_state_dict(self.best_model_weights)
        self._save_model_weights(os.path.join(self.root_dir, 'best_model.pth'))

    @Clock.register(['Testing'])
    def _test_loss(self, dataset, fold: int) -> float:
        """Evaluate the model on a test set and return the loss"""
        device = self.device

        if dataset is None or len(dataset) == 0:
            self.logger.warning('Test set is empty. Skipping test loss evaluation.')
            return -1.0

        transform = ToPILImage()
        num_samples = len(dataset)
        test_loss = 0.0

        test_outdir = f'test_output_fold_{fold + 1}'
        test_outdir = os.path.join(self.root_dir, test_outdir)
        for i, sample in enumerate(dataset):
            pred = self.model(torch.unsqueeze(sample[0],0).to(device),torch.unsqueeze(sample[1],0).to(device),torch.unsqueeze(sample[2],0).to(device))
            test_loss += self.loss_fn(pred, torch.unsqueeze(sample[3], 0).to(device)).item()

            img = transform(torch.squeeze(pred))
            file_out = os.path.join(test_outdir, str(sample[4]))
            self.ensure_dir(file_out)
            img.save(file_out)
            self.logger.debug('Saved %s test output image: %s', i, file_out)

        test_loss /= num_samples
        self.logger.info('Test loss: %.4f', test_loss)

        return test_loss

    def save_losses(self):
        """Save the training, validation, and test losses to a JSON file"""
        losses = {}
        for fold in range(self.num_folds):
            losses[f'fold_{fold + 1}'] = {
                'train_loss': self.train_losses[fold],
                'validation_loss': self.valid_losses[fold],
                'test_loss': self.test_losses[fold],
            }

        losses_file = os.path.join(self.root_dir, 'losses.json')
        with open(losses_file, 'w') as f:
            json.dump(losses, f, indent=4)
