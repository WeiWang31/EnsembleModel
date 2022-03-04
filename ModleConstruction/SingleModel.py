'''
@author: WeiWang
@time: 2021/8/19
@email: ww1119694082@gmail.com
'''
import numpy as np
from keras.layers import Dense, Conv1D, ELU,Flatten, Dropout, BatchNormalization, MaxPooling1D
from keras.models import Model
from keras.layers import Input
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

def single_model():
    input_layer = Input(shape=(4096, 1))

    conv1 = Conv1D(8, kernel_size=64, strides=1)(input_layer)
    elu1 = ELU()(conv1)

    conv2 = Conv1D(8, kernel_size=32, strides=1)(elu1)
    max_pool1 = MaxPooling1D(pool_size=8, strides=8)(conv2)
    elu2 = ELU()(max_pool1)

    conv3 = Conv1D(16, kernel_size=32, strides=1)(elu2)
    elu3 = ELU()(conv3)

    conv4 = Conv1D(16, kernel_size=16, strides=1)(elu3)
    max_pool2 = MaxPooling1D(pool_size=6, strides=6)(conv4)
    elu4 = ELU()(max_pool2)

    conv5 = Conv1D(32, kernel_size=16, strides=1)(elu4)
    elu5 = ELU()(conv5)

    conv6 = Conv1D(32, kernel_size=16, strides=1)(elu5)
    max_pool3 = MaxPooling1D(pool_size=4, strides=4)(conv6)
    elu5 = ELU()(max_pool3)

    flatten = Flatten()(elu5)

    dense1 = Dense(64, activation='elu')(flatten)
    dropout1 = Dropout(0.5)(dense1)

    dense2 = Dense(64, activation='elu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)

    out = Dense(2, activation='softmax')(dropout2)

    model = Model(inputs=input_layer, outputs=out)
    model.compile(optimizer=SGD(lr=0.002, nesterov=True, momentum=0.004), loss='categorical_crossentropy',
                    metrics=['accuracy'])
    # model.summary()
    return model

def trian():
    trian_data = np.load('path of training set')
    train_x = trian_data['strains']
    train_x = train_x.reshape((-1, 4096, 1))
    train_y = trian_data['labels']
    train_y = train_y.reshape((-1, 2))

    check_point = ModelCheckpoint('path to save model weight', monitor='val_accuracy', verbose=1,
                                  save_best_only=True, mode='max')

    model = single_model()
    model.summary()
    history = model.fit(train_x,
                        train_y,
                        batch_size=32,
                        validation_split=0.1,
                        verbose=1,
                        epochs=30,
                        callbacks=[check_point])
    np.savez('path to save data in training process',
             acc=history.history['accuracy'],
             val_acc=history.history['val_accuracy'],
             loss=history.history['loss'],
             val_loss=history.history['val_loss'])

if __name__ == '__main__':
    trian()



