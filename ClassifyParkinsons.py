import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

def loadData():
	file=open("parkinsons/data")
	line=file.readline()
	result=[]

	while(line!=''):
		a=line.split()
		a=map(float,a)
		result.append(a)
		line=file.readline()
	a=np.array(result)
	print a
	return a

def loadTrueClass():
	file=open("parkinsons/trueclass")
	line=file.readline()
	result=[]

	while(line!=''):
		a=line.split()
		a=map(int,a)
		result.append(a)
		line=file.readline()
	a=np.array(result)
	return a

def loadTrainingData(x):
	f=open("Parkinsons/random_class.%d" %x,"r")
	print "------------------------------------------------------------------------------"
	print "Parkinsons/random_class.%d"%x
	line=f.readline()
	result=[]
	
	while(line!=''):
		a=line.split()
		result.append(a)
		line=f.readline()
		
	
	a=np.array(result)
	a=a.astype(int)
	return a

def classify(data, trueclass, traindata, final_set,a):
	X=np.vstack(data[traindata[:,1],:])
	#np.savetxt("parkinsons/foo.csv",x, fmt='%0.5f',delimiter=",")
	b=[]
	b.append(traindata[:,1])
	
	C = np.searchsorted(a, b)
	D = np.delete(np.arange(np.alen(a)), C)
	D= np.array(D)
	D=D.reshape(D.size,-1)
	
	true_labels = np.ravel(np.vstack(trueclass[D[:,0],0]))
	test_data = np.vstack(data[D[:,0],:])
	#print test_data.shape
	#np.savetxt("parkinsons/foo.csv",test_data, fmt='%0.6s')
	y=np.ravel(np.vstack(traindata[:,0]))
	
	clf=svm.SVC(kernel='linear')
	clf.fit(X,y)
	
	labels=clf.predict(test_data) #predicting true labels for the remaining rows 
	predicted_labels = labels.reshape(labels.size,-1)
	np.savetxt("parkinsons/foo%d.csv"%final_set, np.concatenate((test_data, predicted_labels,np.vstack(trueclass[D[:,0],0])), axis=1),fmt='%0.5f',delimiter=",")
	
	print true_labels
	print labels
	misclassify_rate = 1-accuracy_score(true_labels,labels)
	print "Misclassification rate = %f" %misclassify_rate
	return misclassify_rate
	
		
data = loadData()  #loading original data
trueclass = loadTrueClass() #loading true labels
error=0 

for i in range(0,10,1): #looping through 10 training data sets
	traindata=loadTrainingData(i) #loading each training file
	a=[]
	for j in range(0,195,1):
		a.append(j)
			
	misclassify_rate=classify(data, trueclass, traindata,i,a) #classification based on each training set. Also the parameter passing is call by value here
	error = error+misclassify_rate

print error
error = error/10 #average of error across 10 training sets
print "Average Error= %0.3f" %error




