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
        if f and f():
            return True
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
        self.pred = self.model(self.run.item[0])

    def compute_loss(self):
        self.loss = self.run.loss_func(self.pred, self.run.item[1])

    def backward(self):
        self.loss.backward()

    def step(self):
        self.run.opt.step()

    def zero_grad(self):
        self.run.opt.zero_grad()

