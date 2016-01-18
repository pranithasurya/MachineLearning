import sys
import random
import math

datafile = sys.argv[1]
f= open(datafile)
data =[]
i=-0

### Read Data###

l=f.readline()
while(l!=''):
	a=l.split()
	l2=[]
	l2.append(1.0)
	for j in range(0,len(a),1):
		l2.append(float(a[j]))
	data.append(l2)
	l=f.readline()

rows=len(data)
cols=len(data[0])
f.close()

### Read Labels ###
labelfile= sys.argv[2]
f=open(labelfile)
trainlabels ={}
y=[]
X=[]

l=f.readline()
while(l!=''):
	a=l.split()
	if(a[0]=='0'):
		trainlabels[int(a[1])] = -1
	else:
		trainlabels[int(a[1])] = int(a[0])
	y.append(trainlabels[int(a[1])])
	X.append(data[int(a[1])])
	l=f.readline()

#l=list(set(trainlabels.values()))
#print(len(X))
#print(y)
#print(trainlabels)

W=[]
for i in range(0,cols,1):
	W.append(random.random())
#print(W)

### gradiend descent ###
eta = 0.001

#K=list(map(list,zip(*X)))

obj = 0.0
prevobj=float("inf")
wx = []
k=0.0
const=-2
a=0.0
delf=[]

for i in range(rows):
	for j in range(cols):
		#print(W[j],X[i][j])
		k+=W[j]*X[i][j]
	wx.append(0)
	wx[i]=k
	k=0.0
#print(wx)
#print(W)

for i in range(len(y)):
	#print(y[i],wx[i])
	obj += (y[i] - wx[i])**2
#print(obj)

hx=[]
for i in range(rows):
	hx.append(0.0)

for i in range(cols):
	delf.append(0.0)

while(prevobj-obj>0.001):
	prevobj=obj
	#print('obj=',obj)
	for i in range(rows):
		#print(y[i],wx[i])
		hx[i]=(y[i]-wx[i])
	#print(hx)	
	for j in range(cols):
		for i in range(rows):
			#print(i,j,X[i][j],hx[i])
			k+=const*X[i][j]*hx[i]
		delf[j]=k
		k=0.0
	#print(delf)

	for i in range(len(W)):
		#print('w',W[i],'del',delf[i])
		W[i]=W[i]-eta*delf[i]
	#print(W)

	for i in range(rows):
		for j in range(cols):
			#print(i,j,W[j],X[i][j])
			k+=W[j]*X[i][j]
		#print(k)	
		wx[i]=k
		k=0.0
	obj=0.0
	for i in range(len(y)):
		obj+=(y[i]-wx[i])**2
	#print('obj2',obj)
		
print("Weight vector=",W)
sum = 0.0
for i in range(len(W)):
	#print(i)
	sum+=(W[i])**2
W0=math.sqrt(sum)
q=abs(W[0]/W0)
print("Distance to the origin=",q)

