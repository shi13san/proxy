import numpy as np
import time

from discriminator import RandomForrest
from discriminator import LR
from discriminator import DT
from discriminator import NB
from discriminator import SVM
from discriminator import KNN
from discriminator import MLP
from discriminator import VOTE

data = np.loadtxt('../MalGAN/data/API_truncation50_random_split_trainval_1gram_feature.csv',
                               delimiter=',', dtype=np.int32)
test_data = np.loadtxt('../MalGAN/data/API_truncation50_random_split_test_1gram_feature.csv',
                               delimiter=',', dtype=np.int32)
train_data = data[:200]
test_data = data[10000:20000]
times = 100
tpr = 0.0
fpr = 0.0
accuracy = 0.0
auc = 0.0
score_template = 'TPR %(TPR)f\tFPR %(FPR)f\tAccuracy %(Accuracy)f\tAUC %(AUC)f'

for i in range(times):
	D = RandomForrest()
	D.train(train_data[:, :-1], train_data[:, -1])
	res = D.evaluate(test_data[:, :-1], test_data[:, -1])
	tpr += res['TPR']
	fpr += res['FPR']
	accuracy += res['Accuracy']
	auc += res['AUC']
	#print 'RandomForrest: ' + message + '\n'
	# print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
message = score_template % {'TPR':tpr/times, 'FPR':fpr/times, 'Accuracy':accuracy/times, 'AUC':auc/times}
print 'RandomForrest: ' + message + '\n'

tpr = 0.0
fpr = 0.0
accuracy = 0.0
auc = 0.0
for i in range(times):
	D = LR()
	D.train(train_data[:, :-1], train_data[:, -1])
	res = D.evaluate(test_data[:, :-1], test_data[:, -1])
	tpr += res['TPR']
	fpr += res['FPR']
	accuracy += res['Accuracy']
	auc += res['AUC']
message = score_template % {'TPR':tpr/times, 'FPR':fpr/times, 'Accuracy':accuracy/times, 'AUC':auc/times}
print 'LR: ' + message + '\n'

tpr = 0.0
fpr = 0.0
accuracy = 0.0
auc = 0.0
for i in range(times):
	D = DT()
	D.train(train_data[:, :-1], train_data[:, -1])
	res = D.evaluate(test_data[:, :-1], test_data[:, -1])
	tpr += res['TPR']
	fpr += res['FPR']
	accuracy += res['Accuracy']
	auc += res['AUC']
message = score_template % {'TPR':tpr/times, 'FPR':fpr/times, 'Accuracy':accuracy/times, 'AUC':auc/times}
print 'DT: ' + message + '\n'

tpr = 0.0
fpr = 0.0
accuracy = 0.0
auc = 0.0
for i in range(times):
	D = NB()
	D.train(train_data[:, :-1], train_data[:, -1])
	res = D.evaluate(test_data[:, :-1], test_data[:, -1])
	tpr += res['TPR']
	fpr += res['FPR']
	accuracy += res['Accuracy']
	auc += res['AUC']
message = score_template % {'TPR':tpr/times, 'FPR':fpr/times, 'Accuracy':accuracy/times, 'AUC':auc/times}
print 'NB: ' + message + '\n'

tpr = 0.0
fpr = 0.0
accuracy = 0.0
auc = 0.0
for i in range(times):
	D = SVM()
	D.train(train_data[:, :-1], train_data[:, -1])
	res = D.evaluate(test_data[:, :-1], test_data[:, -1])
	tpr += res['TPR']
	fpr += res['FPR']
	accuracy += res['Accuracy']
	auc += res['AUC']
message = score_template % {'TPR':tpr/times, 'FPR':fpr/times, 'Accuracy':accuracy/times, 'AUC':auc/times}
print 'SVM: ' + message + '\n'

tpr = 0.0
fpr = 0.0
accuracy = 0.0
auc = 0.0
for i in range(times):
	D = KNN()
	D.train(train_data[:, :-1], train_data[:, -1])
	res = D.evaluate(test_data[:, :-1], test_data[:, -1])
	tpr += res['TPR']
	fpr += res['FPR']
	accuracy += res['Accuracy']
	auc += res['AUC']
message = score_template % {'TPR':tpr/times, 'FPR':fpr/times, 'Accuracy':accuracy/times, 'AUC':auc/times}
print 'KNN: ' + message + '\n'

tpr = 0.0
fpr = 0.0
accuracy = 0.0
auc = 0.0
for i in range(times):
	D = MLP()
	D.train(train_data[:, :-1], train_data[:, -1])
	res = D.evaluate(test_data[:, :-1], test_data[:, -1])
	tpr += res['TPR']
	fpr += res['FPR']
	accuracy += res['Accuracy']
	auc += res['AUC']
message = score_template % {'TPR':tpr/times, 'FPR':fpr/times, 'Accuracy':accuracy/times, 'AUC':auc/times}
print 'MLP: ' + message + '\n'

tpr = 0.0
fpr = 0.0
accuracy = 0.0
auc = 0.0
for i in range(times):
	D = VOTE()
	D.train(train_data[:, :-1], train_data[:, -1])
	res = D.evaluate(test_data[:, :-1], test_data[:, -1])
	tpr += res['TPR']
	fpr += res['FPR']
	accuracy += res['Accuracy']
	auc += res['AUC']
message = score_template % {'TPR':tpr/times, 'FPR':fpr/times, 'Accuracy':accuracy/times, 'AUC':auc/times}
print 'VOTE: ' + message + '\n'
