from time import time

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_entire_dataset(dataset, collator_fn=None):
    """Load the entire dataset as a tensor"""
    all_x = []
    all_y = []

    for x, y in dataset:
        all_x.append(x)
        all_y.append(y)

    print(f'Single x shape {x.shape}')
    print(f'Single y shape {y.shape}')

    if collator_fn is not None:
        print('Using collator fn')
        all_x, all_y = collator_fn(list(zip(all_x, all_y)))
    else:
        def stack(arr):
            if isinstance(arr, torch.Tensor):
                return torch.stack(arr)
            return np.stack(arr)

        all_x = stack(all_x)
        all_y = stack(all_y)
    print(f"All stack x shape = {all_x.shape}")
    print(f"All stack y shape = {all_y.shape}")
    return all_x, all_y


class OptionalScaler:
    def __init__(self, low=0, high=2, scaler='minmax'):
        self.mms = None
        self.low = low
        self.high = high
        self.scaler = scaler

    @property
    def is_fitted(self):
        return self.mms is not None

    def fit(self, data, tag=None):
        m, M = data.min(), data.max()
        print(f"Min for {tag} = {m}, Max for {tag} = {M}")
        if m < self.low or M > self.high:
            print(f'=== Using {self.scaler} scaler on {tag} ===')
            self.mms = MinMaxScaler() if self.scaler == 'minmax' else StandardScaler()
            self.mms.fit(data.reshape(data.shape[0], -1))
        else:
            print(f"Not scaling {tag}")

    def scale(self, data):
        if self.is_fitted:
            ori_shape = data.shape
            return torch.from_numpy(self.mms.transform(
                data.reshape(data.shape[0], -1)
            ).reshape(ori_shape)).float()
        return data

    def inverse_scale(self, data):
        if self.is_fitted:
            ori_shape = data.shape
            return torch.from_numpy(self.mms.inverse_transform(
                data.reshape(data.shape[0], -1)
            ).reshape(ori_shape)).float()
        return data


class TimeCop:
    def __init__(self, budget=None):
        self.budget = budget
        self.global_starting_time = time()
        self.epoch_durations = []

        # In case the user forgets register_epoch_start
        self.epoch_starting_time = self.global_starting_time

    def register_epoch_start(self):
        self.epoch_starting_time = time()

    def register_epoch_end(self):
        self.epoch_ending_time = time()
        duration = self.epoch_ending_time - self.epoch_starting_time
        self.epoch_durations.append(duration)

    def stop(self) -> bool:
        if self.budget is None or len(self.epoch_durations) == 0:
            return False

        current_time = time()
        time_so_far = current_time - self.global_starting_time
        average_epoch_duration = np.mean(self.epoch_durations)
        remaining_time = self.budget - time_so_far

        # stop if less than 15 mins
        if remaining_time < 900:
            return True

        average_n_epochs_left = remaining_time // average_epoch_duration
        print(f'Average n epochs left: {average_n_epochs_left}')
        # quit if 10 epochs left and the model has been trained for
        # sufficient epochs
        if average_n_epochs_left < 20 and len(self.epoch_durations) >= 25:
            return True
        # if epochs take a long time, give a bit more slack
        elif average_n_epochs_left <= 7:
            return True

        return False


class EarlyStopping:
    def __init__(self, patience=30, min_epochs=150):
        self.patience = patience
        self.min_epochs = min_epochs
        self.inc_loss_count = 0
        self.best_loss = np.inf
        self.epoch_count = 0

    def stop(self, loss) -> bool:
        self.epoch_count += 1

        if loss > self.best_loss:
            self.inc_loss_count += 1
        else:
            self.inc_loss_count = 0
            self.best_loss = loss

        return self.inc_loss_count >= self.patience and self.epoch_count >= self.min_epochs


def something_is_off(loss_list, off_count=20):
    """Raise error if the top 'off_count' losses are strictly increasing
    """
    if len(loss_list) < off_count:
        return False

    for i in range(1, off_count):
        if loss_list[i] < loss_list[i - 1]:
            return False

    return True


class SomethingIsOffException(Exception):
    pass
