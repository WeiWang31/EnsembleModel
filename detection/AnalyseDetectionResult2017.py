'''
@author: WeiWang
@time: 2021/9/20
@email: ww1119694082@gmail.com
'''

from itertools import groupby
import numpy as np
import os


# 这个方法用于获取nan数段中间隔的数段范围
def cut_list_process(lst):
    postion_line = []
    fun = lambda x: x[1] - x[0]
    for k, g in groupby(enumerate(lst), fun):
        l1 = [j for i, j in g]  # 连续数字的列表
        if len(l1) > 1:
            scop = str(min(l1)) + '-' + str(max(l1))  # 将连续数字范围用"-"连接
        else:
            scop = l1[0]
        postion_line.append(scop)
    return postion_line


def conclude_trigger(np_result_list, np_result_list2, h_threshold=0, l_threshold=0, name=''):
    np_result_list[np_result_list >= h_threshold] = 1
    np_result_list[np_result_list < h_threshold] = 0

    np_result_list2[np_result_list2 >= l_threshold] = 1
    np_result_list2[np_result_list2 < l_threshold] = 0

    w_list = np_result_list + np_result_list2

    w_list[w_list < 2] = 0
    w_list[w_list == 2] = np.NAN

    trigger_list = np.zeros(5)
    result = np.where(np.isnan(w_list))[0]

    # get the combined result of whole ensemble model
    result = cut_list_process(result)

    for i in result:
        if type(i) != str:
            trigger_list[0] = trigger_list[0] + 1
        else:
            start_end = i.split('-')
            trigger_len = int(start_end[1]) - int(start_end[0]) + 1
            if trigger_len >= 5:
                trigger_list[4] = trigger_list[4] + 1
                gps_start = float(name.split('-')[0])
                alarm_start = float(i.split('-')[0])
                alarm_end = float(i.split('-')[1])
                merge_predict = ((alarm_start / 8.0 + 1) + alarm_end / 8) / 2.0 + gps_start
                print('merge time prediction：' + str(merge_predict))

            else:
                trigger_list[trigger_len-1] = trigger_list[trigger_len-1] + 1

    return trigger_list



if __name__ == '__main__':
    # read the detection result of Hanford and Livingston respectively
    # Note that the result is generated by mode III

    h_path = '../detection_result_August_2017/2017_h/'
    l_path = '../detection_result_August_2017/2017_l/'
    file_list = os.listdir(h_path)
    file_list.sort()

    h_trigger = np.zeros(50)
    l_trigger = np.zeros(50)
    for i in range(len(file_list)):
        name = file_list[i].split('_')[1]

        h_name = h_path + 'h_' + name
        l_name = l_path + 'l_' + name
        h_result = np.load(h_name)
        l_result = np.load(l_name)

        # input the threshold
        h_l_list = conclude_trigger(np_result_list=h_result,
                         np_result_list2=l_result,
                         h_threshold=0.8343,
                         l_threshold=0.8454,
                         name=name.split('.n')[0])
        if h_l_list[-1] > 0:
            print('GPS: ' + name.split('.n')[0])
            print(h_l_list)