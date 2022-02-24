import numpy as np
from SingleModel import single_model
from sklearn import metrics

def get_AUC_list():
    test_set = np.load('path of test set')
    test_x = test_set['strains']
    test_x = test_x.reshape((-1, 4096, 1))
    test_y = test_set['labels']
    test_y = test_y.reshape((-1, 2))

    AUC_list = []

    for i in range(100):
        model = single_model()
        model.load_weights('path to save weight of weak learners/' + 'hb' + str(i) + '.h5')
        result_list = model.predict(test_x)
        m_fa, m_ta, _ = metrics.roc_curve(test_y[:, 0], result_list)
        m_auc = metrics.auc(m_fa, m_ta)
        AUC_list.append(m_auc)

    # output the top 10 weak learner's id
    AUC_list = np.array(AUC_list)
    print(np.argsort(-AUC_list) + 1)


