from keras.callbacks import Callback

class TerminateTrainingLoss(Callback):

    """
    Termina o treino quando atinge determinada loss
    """

    def __init__(self, minimal_loss=0.2):
        self.minimal_loss = minimal_loss

    def on_epoch_end(self, epoch, logs=None, verbose=False):
        
        if logs.get('loss') < self.minimal_loss:
            self.model.stop_training = True

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        # For backwards compatibility
        self.on_batch_begin(batch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        # For backwards compatibility
        self.on_batch_end(batch, logs=logs)

    def on_train_end(self, logs=None):
        pass