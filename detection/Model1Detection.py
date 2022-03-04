'''
@author: WeiWang
@time: 2021/9/20
@email: ww1119694082@gmail.com
'''
from keras.optimizers import SGD
from ModleConstruction.SingleModel import single_model
import numpy as np
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

# model1 detect O1 and O2 events in 16s
def detection(name, type):
    test_data = np.load('../TrueEvent/16s_0.125s/GW'+ name + '.npz')
    if type == 'h':
        test_data = test_data['h_strain'].reshape(-1, 4096, 1)
    elif type == 'l':
        test_data = test_data['l_strain'].reshape(-1, 4096, 1)

    model = single_model()
    model.compile(optimizer=SGD(lr=0.002, nesterov=True, momentum=0.004), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    if type == 'h':
        model.load_weights('../h5/Hanford/hmodel1.h5')
    if type == 'l':
        model.load_weights('../h5/Livingston/lmodel1.h5')

    result_list = model.predict(test_data)
    np_result_list = result_list[:, 0]
    real_time_detect(np_result_list, name, type)

if __name__ == '__main__':
    names = ['150914', '151012', '151226', '170104', '170608', '170729', '170809', '170814', '170817', '170818',
             '170823']

    for name in names:
        detection(type='h', name=name)

