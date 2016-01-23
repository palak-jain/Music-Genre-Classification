from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
import matplotlib.pyplot as pl
import numpy as np
df = ClassificationDataSet(3,class_labels=['pop','classical'])
file1 = open('dummy2.csv','r')

for line in file1.readlines():
	data = [x for x in line.strip().split(', ') if x != '']
	indata = tuple(data[:3])
	outdata = tuple(data[3:4])
	df.addSample(indata,outdata)
trndata, tstdata = df.splitWithProportion( 0.75 )
print(trndata)
# tstdata, validata = df.splitWithProportion( 0.50 )
# trndata._convertToOneOfMany()
# tstdata._convertToOneOfMany()
# print(trndata.indim,trndata.outdim)
n = buildNetwork(df.indim,1,df.outdim,outclass=SoftmaxLayer )
t = BackpropTrainer(n,dataset=trndata,momentum=0.1,verbose=True,weightdecay=0.01)
trnerr=t.trainUntilConvergence(verbose=True)
# pl.plot(trnerr,'b',valerr,'r')
# t.trainOnDataset(trndata,99)
out=t.testOnClassData(tstdata)
# out=out.argmax(axis=1)
# out=out.reshape(indata.shape)
# print(out.shape)
print(out)
# output=np.array([n.activate(x) for x,_ in validata])
# output=output.argmax(axis=1)
# print(output)