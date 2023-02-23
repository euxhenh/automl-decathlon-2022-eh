"""An example of code submission for the AutoML Decathlon challenge.

It implements 3 compulsory methods ('__init__', 'train' and 'test') and an
attribute 'done_training' for indicating if the model will not proceed more
training due to convergence or limited time budget.

To create a valid submission, zip model.py and metadata together with other
necessary files such as tasks_to_run.yaml, Python modules/packages,
pre-trained weights, etc. The final zip file should not exceed 300MB.
"""

import logging
import sys
import traceback

import numpy as np
import torch
from convgru import ConvGRU
from gru_rnn import MyGRU
from model_wrn import get_wrn
from model_xgb import Model as ModelXGB
from trainer import Trainer

# seeding randomness for reproducibility
np.random.seed(42)
torch.manual_seed(1)


def get_hidden_size(nontime_dims):
    if nontime_dims <= 128:
        hidden_size = 100
    elif nontime_dims <= 512:
        hidden_size = 256
    else:
        hidden_size = 512
    print(f"Choosing {hidden_size=}")
    return hidden_size


def get_batch_size(n_samples):
    n_root = np.sqrt(n_samples)
    if n_root <= 20:
        batch_size = 16
    elif n_root <= 32:
        batch_size = 32
    elif n_root <= 64:
        batch_size = 64
    else:
        batch_size = 128
    print(f"Choosing {batch_size=}")
    return batch_size


def get_combo_model(metadata):
    """
    GRU if time dimension and not 2 space dims
    Conv GRU if time dimension and 2 space dims
    XGB if no spacetime dims
    WRN otherwise
    """
    row_count, col_count = metadata.get_tensor_shape()[2:4]
    channel = metadata.get_tensor_shape()[1]
    sequence_size = metadata.get_tensor_shape()[0]
    input_shape = (sequence_size, channel, row_count, col_count)

    output_size = np.prod(metadata.get_output_shape())
    print(f"Outputing {output_size} dimensions")

    spacetime_dims = np.count_nonzero(np.array(input_shape)[[0, 2, 3]] != 1)
    nontime_dims = np.prod(np.array(input_shape)[[1, 2, 3]])
    scaler = 'minmax'
    low = -5
    high = 5

    if sequence_size > 1:
        if row_count == 1 or col_count == 1:
            # Use linear gru if not an image
            print("========== GRU ==========")
            model = MyGRU(
                input_size=nontime_dims,
                hidden_size=get_hidden_size(nontime_dims),
                num_layers=2,
                output_dim=output_size,
            )
        else:
            print("========= Conv GRU =========")
            has_gpu = torch.cuda.is_available()
            model = ConvGRU(
                input_size=(row_count, col_count),
                input_dim=channel,
                hidden_dim=[32, 1],
                kernel_size=(3, 3),
                num_layers=2,
                batch_first=True,
                output_size=output_size,
                dtype=torch.cuda.FloatTensor if has_gpu else torch.FloatTensor,
            )
        # return SafeGRU(metadata, trainer)
    elif spacetime_dims == 0:
        print("=========== XGB ===========")
        model = ModelXGB(metadata)
        return model
    else:
        print("=========== WRN ==========")
        model = get_wrn(
            input_shape=input_shape,
            output_dim=output_size,
            output_shape=metadata.get_output_shape(),
            in_channels=channel,
            depth=16,
            widen_factor=4,
            dropRate=0.0,
        )
        scaler = 'standard'

    trainer = Trainer(
        model=model,
        n_epochs=300,
        lr=3e-3,
        min_epochs=270,
        weight_decay=3e-4,
        task_type=metadata.get_task_type(),
        batch_size=get_batch_size(metadata.size()),
        scaler=scaler,
        low=low,
        high=high,
    )
    return trainer


class SafeGRU:
    def __init__(self, metadata, trainer):
        self.metadata = metadata
        self.trainer = trainer

    def train(
        self,
        dataset,
        val_dataset=None,
        val_metadata=None,
        remaining_time_budget=None
    ):
        try:
            self.trainer.train(
                dataset, val_dataset, val_metadata, remaining_time_budget)
        except:
            traceback.print_exc()
            # Resort to XGB in case of error
            print("Error occured in GRU. Switching to XGBoost.")
            self.trainer = ModelXGB(self.metadata)
            self.trainer.train(
                dataset, val_dataset, val_metadata, remaining_time_budget)

    def test(
        self, dataset, remaining_time_budget=None
    ):
        return self.trainer.test(dataset, remaining_time_budget)


class Model:
    def __init__(self, metadata):
        """
        The initalization procedure for your method given the metadata of
        the task
        Args:
          metadata: an DecathlonMetadata object. Its definition can be
          found in
              ingestion/dev_datasets.py
        """
        # Creating model
        self.model = get_combo_model(metadata=metadata)

    def train(
        self,
        dataset,
        val_dataset=None,
        val_metadata=None,
        remaining_time_budget=None
    ):
        """
        The training procedure of your method given training data,
        validation data (which is only directly provided in certain tasks,
        otherwise you are free to create your own validation strategies),
        and remaining time budget for training.
        """

        """Train this algorithm on the Pytorch dataset.
        ****************************************************************
        ****************************************************************
        Args:
          dataset: a `DecathlonDataset` object. Each of its examples is of
          the form
                (example, labels)
              where `example` is a dense 4-D Tensor of shape
                (sequence_size, row_count, col_count, num_channels)
              and `labels` is a 1-D or 2-D Tensor
          val_dataset: a 'DecathlonDataset' object. Is not 'None' if a
          pre-split validation set is provided, in which case you should
          use it for any validation purposes. Otherwise, you are free to
          create your own validation split(s) as desired.

          val_metadata: a 'DecathlonMetadata' object, corresponding to
          'val_dataset'. remaining_time_budget: time remaining to execute
          train(). The method
              should keep track of its execution time to avoid exceeding
              its time budget. If remaining_time_budget is None, no time
              budget is imposed.

          remaining_time_budget: the time budget constraint for the task,
          which may influence the training procedure.
        """

        return self.model.train(
            dataset, val_dataset, val_metadata, remaining_time_budget
        )

    def test(self, dataset, remaining_time_budget=None):
        """Test this algorithm on the Pytorch dataloader.
        Args:
          Same as that of `train` method, except that the `labels` will be
          empty.
        Returns:
          predictions: A `numpy.ndarray` matrix of shape (sample_count,
          output_dim).
              here `sample_count` is the number of examples in this dataset
              as test set and `output_dim` is the number of labels to be
              predicted. The values should be binary or in the interval
              [0,1].
          remaining_time_budget: the remaining time budget left for
          testing, post-training
        """

        return self.model.test(dataset, remaining_time_budget)

    ##############################################################################
    #### Above 3 methods (__init__, train, test) should always be implemented ####
    ##############################################################################


def get_logger(verbosity_level):
    """Set logging format to something like:
    2019-04-25 12:52:51,924 INFO model.py: <message>
    """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(filename)s: %(message)s"
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


logger = get_logger("INFO")
