class DataBunch:
    def __init__(self, train_dl, valid_dl=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
