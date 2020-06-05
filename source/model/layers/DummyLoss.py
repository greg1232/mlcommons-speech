import tensorflow as tf

class DummyLoss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO,
                 name='dummy'):
        super(DummyLoss, self).__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        return y_pred


