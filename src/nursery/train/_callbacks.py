class Callback:
    """
    Mother class of all the callbacks. It serves three purposes:
    - __call__ execute the function
    - set_runner associate the runner to the callback so it can access the runners' attributes
    - __getattr__ fetch the attribute in the runner class if it is not found in the callback
    Sets also by default the _order attribute to 0. The callbacks are executed by the runner
    in a sorted manner with respect to the value of the _order attribute.
    """

    _order = 0

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        # print(self.__class__.__name__)
        # print(cb_name)
        if f and f():
            # print("True")
            return True
        # print("False")
        return False

    def set_runner(self, run):
        self.run = run

    def __getattr__(self, k):
        return getattr(self.run, k)


class TrainEvalCallback(Callback):
    """
    Mandatory callback to include. Specify which epoch we are currently at in a float format
    and set the model in train or eval mode.
    """

    _order = 0

    def __repr__(self):
        return "TrainEvalCallback \n"

    def begin_fit(self):
        self.run.n_epochs = 0.0

    def after_batch(self):
        if not self.in_train:
            return
        self.run.n_epochs += 1.0 / self.n_iters

    def begin_epoch(self):
        self.run.n_epochs = self.epoch
        self.model.train()
        self.run.in_train = True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train = False


class TrainModel(Callback):
    """Callback to train the network
    """

    _order = 10

    def forward_pass(self):
        self.run.pred = self.model(self.run.item[0])

    def compute_loss(self):
        self.run.loss_val = self.run.loss_func(self.run.pred, self.run.item[1])

    def backward(self):
        self.run.loss_val.backward()

    def step(self):
        self.run.opt.step()

    def zero_grad(self):
        self.run.opt.zero_grad()


class PrintInfo(Callback):
    _order = 15

    def compute_loss(self):
        print(
            f"Epoch: {self.run.n_epochs}, Target: {self.run.item[1]}, Loss: {self.run.loss_val}"
        )


class Novalidation(Callback):
    def begin_validate(self):
        return True
