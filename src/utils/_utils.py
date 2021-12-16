import logging
import time

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


class CancelTrainException(Exception):
    pass


class CancelEpochException(Exception):
    pass


class CancelBatchException(Exception):
    pass


class Learner:
    """
    The Learner class encapsulate the model that will be trained
    as well as the optimizer, the loss function and the data. It does not do
    anything specific but makes the whole code neater.
    """

    def __init__(self, model, opt, data, loss_func=None):
        self.model = model
        self.opt = opt
        self.data = data
        self.loss_func = loss_func


class Runner:
    def __init__(self, cbs: list):
        self.cbs = cbs
        self.stop = False

    @property
    def opt(self):
        return self.learn.opt

    @property
    def model(self):
        return self.learn.model

    @property
    def loss_func(self):
        return self.learn.loss_func

    @property
    def data(self):
        return self.learn.data

    def one_batch(self, item):
        try:
            self.item = item
            self("begin_batch")
            self("forward_pass")
            self("after_pred")
            self("compute_loss")
            self("after_loss")
            if not self.in_train:
                return
            self("backward")
            self("after_backward")
            self("step")
            self("after_step")
            self("zero_grad")

        except CancelBatchException:
            self("after_cancel_batch")

        finally:
            self("after_batch")

    def all_batches(self, dl):
        self.n_iters = len(dl)
        try:
            for itr, item in enumerate(dl):
                self.itr = itr
                self("before_batch")
                self.one_batch(item)

        except CancelEpochException:
            self("after_cancel_epoch")

    def fit(self, epochs, learn):
        self.epochs = epochs
        self.learn = learn
        try:
            for cb in self.cbs:
                cb.set_runner(self)
            self("begin_fit")
            for epoch in range(epochs):
                start = time.time()
                self.epoch = epoch
                if not self("begin_epoch"):
                    self.all_batches(self.data.train_dl)
                self("after_train")
                with torch.no_grad():
                    if not self("begin_validate"):
                        self.all_batches(self.data.valid_dl)
                    self("after_val")
                self("after_epoch")
                logger.info(
                    f"Epoch = {epoch}, Elapsed time {time.time() - start} seconds"
                )

        except CancelTrainException:
            self("after_cancel_train")

        finally:
            self("after_fit")
            # self.learn = None

    def __call__(self, cb_name):
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order):
            res = cb(cb_name) or res
        return res
