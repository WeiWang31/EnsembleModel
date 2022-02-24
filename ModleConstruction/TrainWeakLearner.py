from SingleModel import single_model
import numpy as np
from keras.callbacks import ModelCheckpoint


def load_data(path):
    trian_data = np.load(path)
    train_x = trian_data['strains']
    train_x = train_x.reshape((-1, 4096, 1))
    train_y = trian_data['labels']
    train_y = train_y.reshape((-1, 2))
    return [train_x, train_y]


def batch_train():
    for i in range(100):
        model = single_model()

        # hw represents weak learners for Hanford
        # lw represents weak learners for Livingston

        check_point = ModelCheckpoint('path to save weight of weak learners/hw' + str(i + 1) + '.h5', monitor='val_accuracy', verbose=1,
                                      save_best_only=True, mode='max')
        path = 'path to save augmented training set /ht' + str(i + 1) + '.npz'
        train_data = load_data(path)
        train_x = train_data[0]
        train_y = train_data[1]
        history = model.fit(train_x,
                            train_y,
                            epochs=60,
                            batch_size=32,
                            validation_split=0.1,
                            callbacks=[check_point],
                            verbose=1)

if __name__ == '__main__':
    batch_train()


