class MyPolynomialDecay:
    """
    Class to define a Polynomial Decay for the learning rate
    """
    def __init__(self, max_epochs=100, init_lr=1e-3, power=1.0):
        """
        Class constructor.
        :param max_epochs (int): Max number of epochs
        :param init_lr (float): Initial Learning Rate
        :param power (float): Power
        """
        # store the maximum number of epochs, base learning rate,
        # and power of the polynomial
        self.max_epochs = max_epochs
        self.init_lr = init_lr
        self.power = power

    def __call__(self, epoch):
        """
        Called by the callback.
        :param epoch (int): Current epoch
        :return updated_lr (float): Updated LR
        """
        # compute the new learning rate based on polynomial decay
        decay = (1 - (epoch / float(self.max_epochs))) ** self.power
        alpha = self.init_lr * decay
        # return the new learning rate
        return float(alpha)
