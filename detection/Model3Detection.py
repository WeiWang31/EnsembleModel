'''
@author: WeiWang
@time: 2021/9/20
@email: ww1119694082@gmail.com
'''
from keras.optimizers import SGD
from ModleConstruction.SingleModel import single_model
import numpy as np
from keras import Model
from keras import backend as K
from keras.layers import Average, Lambda
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def real_time_detect(list1, name, type):
    list_len = len(list1)
    x = np.linspace(-8, 8, list_len)
    plt.figure(figsize=(10, 2))
    plt.gcf().subplots_adjust(bottom=0.16)
    ax = plt.gca()
    ax.set_ylim(0, 1)
    ax.set_xlim(-8, 8)
    y_major_locator = MultipleLocator(0.5)
    x_major_locator = MultipleLocator(4)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    if type == 'h':
        plt.title('H-GW' + name)
    elif type == 'l':
        plt.title('L-GW' + name)
    plt.plot(x, list1, linestyle='-')
    plt.grid()
    plt.show()


def optimized_vote(x):
    x = K.ones_like(x) / x
    x = K.mean(x, axis=0)
    x = K.ones_like(x) / x
    return x

# model3 detect O1 and O2 events in 16s
def detection(line, name, type):
    test_data = np.load('../TrueEvent/16s_0.125s/GW'+ name + '.npz')
    if type == 'h':
        test_data = test_data['h_strain'].reshape(-1, 4096, 1)
    elif type == 'l':
        test_data = test_data['l_strain'].reshape(-1, 4096, 1)

    print(test_data.shape)

    test_list = []
    input_layers = []
    output_layers = []

    if type == 'h':
        for i in line:
            test_list.append(test_data)
            model =single_model()
            model.load_weights('../h5/Hanford/hw' + str(i) + '.h5')
            input_layers.append(model.input)
            output_layers.append(model.layers[21].output)

    if type == 'l':
        for i in line:
            test_list.append(test_data)
            model = single_model()
            model.load_weights('../h5/Livingston/lw' + str(i) + '.h5')
            input_layers.append(model.input)
            output_layers.append(model.layers[21].output)

    vote_layer = Lambda(optimized_vote, name='optimized_vote')(output_layers)

    model = Model(inputs=input_layers, outputs=vote_layer)

    model.compile(optimizer=SGD(lr=0.002, nesterov=True, momentum=0.004), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    result_list = model.predict(test_list)
    np_result_list = result_list[:, 0]
    real_time_detect(np_result_list, name, type)


if __name__ == '__main__':
    names = ['150914', '151012', '151226', '170104', '170608', '170729', '170809', '170814', '170817', '170818',
             '170823']
    H1_line = [11, 30, 47, 4, 79, 42, 57, 61, 83, 95]
    L1_line = [38, 32, 87, 92, 35, 4, 12, 17, 30, 81]

    # H1_line represents base leaner for Hanford
    # L1_line represents base leaner for Livingston
    for name in names:
        detection(line=H1_line, type='h', name=name)

