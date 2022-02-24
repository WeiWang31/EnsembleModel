import numpy as np
import random

def aug(id):
    data = np.load('path of sample/sample' + id + '.npz')

    strains = data['strains']
    labels = data['labels']

    # 60000 8s strain augment to 120000 1s strain
    labels_aug = np.concatenate((labels, labels), axis=0)
    strains_aug = []

    for i in range(2):
        for j in range(strains.shape[0]):

            # the merge time of 8s strain is 5.5s
            # random cut the 8s strain to let the merge time locate in (1/8,  7/8) s in 1s strain
            # 18943 = 4096(sapling rate) * 5.5 - 1 - (4096 * 7/8)
            # 22015 = 4096(sapling rate) * 5.5 - 1 - (4096 * 1/8)

            position_left = random.randint(18943, 22015)
            position_right = position_left + 4096
            strains_aug.append(strains[j, position_left:position_right])

    strains_aug = np.array(strains_aug)
    strains_aug = strains_aug.reshape((-1, 4096))

    # ht represents augmented training set for Hanford
    # lt represents augmented training set for Livingston
    np.savez('path to save augmented training set/ht' + str(id),
             strains=strains_aug,
             labels=labels_aug)

if __name__ == '__main__':
    for i in range(100):
        aug(i + 1)










