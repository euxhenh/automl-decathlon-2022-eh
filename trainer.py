import copy
import time
import gc
import warnings

import numpy as np
import torch
import torch.nn as nn
from eh_utils import (EarlyStopping, OptionalScaler, SomethingIsOffException,
                      TimeCop, get_entire_dataset, something_is_off)
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


TASK_TO_LOSS = {
    'continuous': nn.MSELoss,
    'single-label': nn.CrossEntropyLoss,
    'multi-label': nn.BCEWithLogitsLoss,
    # 'multi-label': nn.MultiLabelSoftMarginLoss,
}

TASK_TO_PRED_LAMBDA = {
    'continuous': lambda x: x,  # identity
    'single-label': lambda x: torch.softmax(x, dim=1).data,
    'multi-label': lambda x: torch.sigmoid(x).data,
}


class Trainer:
    def __init__(
        self,
        model,
        lr=3e-3,
        n_epochs=300,
        batch_size=128,
        weight_decay=0.0,
        val_frac=0.3,
        patience=30,
        off_count=10,
        min_epochs=150,
        optimizer=torch.optim.Adam,
        task_type='continuous',
        scaler='minmax',
        low=-5,
        high=5,
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.val_frac = val_frac
        self.patience = patience
        self.off_count = off_count
        self.min_epochs = min_epochs
        self.total_test_time = 0
        self.scaler = scaler
        self.low = low
        self.high = high

        self.optimizer = optimizer(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.task_type = task_type
        self.loss_fn = TASK_TO_LOSS[self.task_type]()
        print(f"Using loss={self.loss_fn.__class__.__name__}")

        self.osx = OptionalScaler(low=low, high=high, scaler=self.scaler)
        self.osy = OptionalScaler(low=low, high=high, scaler=self.scaler)

        self.early_stop = EarlyStopping(self.patience, min_epochs=self.min_epochs)

    def get_dataloader(self, dataset, shuffle=True, req_bs=None, col_fn=None):
        dataloader = DataLoader(
            dataset,
            req_bs or self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            collate_fn=col_fn,
        )
        return dataloader

    def train(
        self,
        dataset,
        val_dataset=None,
        val_metadata=None,
        remaining_time_budget=None,
    ):
        self.time_cop = TimeCop(remaining_time_budget)

        self.req_bs = self.req_bs_val = dataset.required_batch_size
        self.col_fn = self.col_fn_val = dataset.collate_fn

        # need this for since col_fn doesnt work on split datasets
        if val_dataset is None and dataset.collate_fn is None:
            n_samples = len(dataset)
            n_valid_samples = int(n_samples * self.val_frac)
            dataset, val_dataset = random_split(
                dataset,
                [n_samples - n_valid_samples, n_valid_samples]
            )
        elif val_dataset is not None:
            self.req_bs_val = val_dataset.required_batch_size
            self.col_fn_val = val_dataset.collate_fn

        self.use_val = val_dataset is not None

        x_train, y_train = get_entire_dataset(dataset, self.col_fn)
        # fit scalers
        self.osx.fit(x_train, 'x')
        self.osy.fit(y_train, 'y')
        del x_train
        del y_train
        gc.collect()

        train_loader = self.get_dataloader(
            dataset,
            req_bs=self.req_bs,
            col_fn=self.col_fn,
        )
        if self.use_val:
            val_loader = self.get_dataloader(
                val_dataset,
                shuffle=False,
                req_bs=self.req_bs_val,
                col_fn=self.col_fn_val,
            )

        self.val_losses = []
        best_loss = np.inf
        best_model = None

        print("Training...")
        for epoch in tqdm(range(self.n_epochs), desc="Epochs"):
            self.time_cop.register_epoch_start()

            self.model.train()
            epoch_losses = []

            for x_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                x_batch, y_batch = self.prep_batch(x_batch, y_batch)
                yhat = self.model(x_batch)
                loss = self.loss_fn(yhat, y_batch.reshape(yhat.shape))
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item() * len(x_batch))

            train_loss = np.sum(epoch_losses) / len(train_loader.dataset)

            if self.use_val:
                val_loss = self.validate(val_loader)
            else:
                val_loss = train_loss

            print(
                f"Epoch {epoch + 1}/{self.n_epochs} :: "
                f"Train Loss {train_loss} :: "
                f"Val Loss {val_loss}"
            )

            if best_loss > val_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(self.model.state_dict())
                print("Updating best model weights...")

            self.val_losses.append(val_loss)

            self.time_cop.register_epoch_end()

            if self.time_cop.stop():
                print(f"TimeCop says stop! Trained for {epoch} epochs.")
                break
            if self.early_stop.stop(val_loss):
                print(f"Early stopping...")
                break

        # Load best weights
        print(f"Loading model with {best_loss=}")
        self.model.load_state_dict(best_model)

    def validate(self, val_loader) -> float:
        if not self.use_val:
            return None

        with torch.no_grad():
            epoch_losses = []
            self.model.eval()

            for x_batch, y_batch in val_loader:
                x_batch, y_batch = self.prep_batch(x_batch, y_batch)
                yhat = self.model(x_batch)
                loss = self.loss_fn(yhat, y_batch.reshape(yhat.shape))
                epoch_losses.append(loss.item() * len(x_batch))

            loss = np.sum(epoch_losses) / len(val_loader.dataset)
            return loss

    def test(
        self,
        dataset,
        remaining_time_budget=None,
    ):
        test_begin = time.time()
        test_loader = self.get_dataloader(
            dataset,
            shuffle=False,
            req_bs=dataset.required_batch_size,
            col_fn=dataset.collate_fn,
        )

        print("Testing...")
        preds = []
        with torch.no_grad():
            self.model.eval()

            for x, _ in iter(test_loader):
                x, _ = self.prep_batch(x)
                pred = self.model(x)
                pred = TASK_TO_PRED_LAMBDA[self.task_type](pred)
                pred = self.osy.inverse_scale(pred.cpu() if pred.is_cuda else pred)
                pred = (pred.cpu() if pred.is_cuda else pred).numpy()
                preds.append(pred)

        preds = np.vstack(preds)

        test_end = time.time()
        test_duration = test_end - test_begin
        self.total_test_time += test_duration
        print(
            "[+] Successfully made one prediction. {:.2f} sec used. ".format(
                test_duration
            )
            + "Total time used for testing: {:.2f} sec. ".format(self.total_test_time)
        )
        return preds

    def prep_batch(self, x_batch, y_batch=None):
        x_batch = x_batch.float()
        x_batch = self.osx.scale(x_batch)

        if y_batch is not None:
            y_batch = y_batch.float()
            y_batch = self.osy.scale(y_batch)
            y_batch = y_batch.to(self.device)

        return x_batch.to(self.device), y_batch
