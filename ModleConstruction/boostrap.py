import numpy as np

def bootstrap(sample_id):
    # The original training set contain 60000 sample data (30000 GW, 30000 noise)
    train_data = np.load('path of original training set')
    train_x = train_data['l1_strains']
    train_y = train_data['l1_labels']

    # Divide the data set into 600 parts, which parts contain 100 sample data
    label_list = np.split(train_y, 600, axis=0)
    strain_list = np.split(train_x, 600, axis=0)

    # Randomly selected sequence, num belong to [0, 599]
    random_list = np.random.randint(0, 600, 600).tolist()

    # begin bootstrap
    strains_1 = strain_list[random_list[0]]
    labels_1 = label_list[random_list[0]]

    for j in range(1, 600):
        print(j)
        labels_1 = np.concatenate([labels_1, label_list[random_list[j]]], axis=0)
        strains_1 = np.concatenate([strains_1, strain_list[random_list[j]]], axis=0)

    np.savez('sample' + str(sample_id),
             strains=strains_1,
             labels=labels_1)

if __name__ == '__main__':
    for i in range(100):
        bootstrap(sample_id=i)

