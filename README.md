# EnsembleModel
本集成模型由江西理工大学与北京师范大学联合设计。

该项目为论文《Ensemble of Deep Convolutional Neural Networks for real-time gravitational wave signal recognition》实验中集成模型的构造代码。

项目不仅公开了模型构造代码，还公开了论文中集成模型中所有的基模型的权重和16s的O1、O2事件的白化应变信号。研究者可以根据以上数据，直接运行model1Detect、model2Detect、model3Detect、来复现论文中16s的探测结果。由于2017年8月的白化探测数据过大，我们仅公开集成模型对该时间段数据的探测结果（位于detection_result_August_2017文件夹内），研究者可以运行AnalyseDetectionResult2017.py来复现总集成模型的探测结果，和对于2017年8月BBH时间的合并时间预测。

研究者还可以根据以上代码构建自己的集成模型对引力波进行探测。欢迎研究者与我们共同交流与探讨。

The ensemble model is designed by the joint efforts of Jiangxi University of Science and Technology and Beijing Normal University.

This project is the construction code of the ensemble model in the paper 'Ensemble of Deep Convolutional Neural Networks for real-time gravitational wave signal recognition'

The project not only exposes the model construction code, but also the weights of all the base learners in the ensemble model and the whitening 16s strain of O1 and O2 events in the paper. Based on the above data, researchers can directly run Model1Detection, Model2Detection, Model3Detection to reproduce the 16S detection results in the paper. Since the whiten strain of August 2017 is too large, we only expose the ensemble model's detection results for that time period (located in the detection_result_August_2017), and the researchers can run AnalyseDetectionResult2017.py to reproduce the detection results of the whole ensemble model and predict the combined time of BBH events in August 2017.

Researchers can also build their own ensemble models to detect gravitational waves from the above code. We welcome all researcher discuss with us.
