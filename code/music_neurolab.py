import pandas as pd;
import neurolab as nl;
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

data=pd.read_csv('./genres/inp_pop_metal.txt')
mn=[]
mx=[]
inp_col=52
out_col=2
col=54
row=out_col*100
hidden=12
max_iter=10
i=1

out=data.ix[:,col-out_col:col]
data=data.ix[:,:inp_col]

for i in range(1,inp_col+1):
    m=[min(data[str(i)+" "]),max(data[str(i)+" "])]
    mn.append(m)

train_in,test_in=[],[]
train_target,test_target=[],[]
train_out,test_out=[],[]
net = nl.net.newff(mn,[hidden,out_col])

for i in range(0,row):
    m=0
    for j in range(1,out_col):
        if out.ix[i,m]<out.ix[i,j]:
            m=j   
    if(i%4==0):
        test_in.append(data.loc[i])
        test_out.append(out.loc[i])
        test_target.append(m)  
    else:
        train_in.append(data.loc[i])
        train_out.append(out.loc[i])
        train_target.append(m)
    
#reshaping
train_in=np.asarray(train_in)
train_in=train_in.reshape(len(train_in),inp_col)
test_in=np.asarray(test_in)
test_in=test_in.reshape(len(test_in),inp_col)
train_out=np.asarray(train_out)
train_out=train_out.reshape(len(train_out),out_col)
test_out=np.asarray(test_out)
test_out=test_out.reshape(len(test_out),out_col)

# print(np.shape(train_in),np.shape(train_out))
# error=nl.net.train.train_gdx(net, train_in, train_out,epochs=1000, show=100, goal=0.1, lr=0.05)

# Train network max_iter times at MAX
it=0
train_error=[100.0]
while (it<max_iter and train_error[-1]>10.0):
    train_error = net.train(train_in, train_out, epochs=1000, show=100, goal=0.1)
    if len(train_error)==0:
        print("error empty ",len(train_error))
        break
    it=it+1


# print(np.shape(train_error))
# Simulate network
out = net.sim(train_in)
out2 = net.sim(test_in)

x = np.linspace(1,row/4 ,row/4)
x.reshape(row/4,1)

train_output,test_output=[],[]
for i in range(0,len(out2)):
    m=0
    for j in range(1,out_col):
        if out2[i,m]<out2[i,j]:
            m=j   
    test_output.append(m)
    
for i in range(0,len(out)):
    m=0
    for j in range(1,out_col):
        if out[i,m]<out[i,j]:
            m=j   
    train_output.append(m)

test_error=0
selectivity,specificty=[],[]
for j in range(0,out_col):
    fn,fp,tp,tn=0,0,0,0
    for i in range(0,len(test_target)):
        fn+=(test_output[i]!=j and test_target[i]==j)
        fp+=(test_output[i]==j and test_target[i]!=j)
        tp+=(test_output[i]==j and test_target[i]==j)
        tn+=(test_output[i]!=j and test_target[i]!=j)
    specificty.append(tp/(tp+fn))
    selectivity.append(tn/(tn+fp))
    # test_error+=(test_target[i]-test_output[i])**2

# print(test_output)
# print(test_target)
print("training error ",train_error[-1])
# print("test error over ",len(test_target), "test inputs ",test_error)
print("selectivity ",selectivity)
print("specificty",specificty)

# Plot result
plt.plot(train_error)
plt.ylabel("error")
plt.xlabel("epochcs")
plt.show()

plt.scatter(x, test_target,color='blue')
plt.scatter(x, test_output,color='red')

# plt.scatter(x, out2[:,1],color='cyan')
# plt.scatter(x, out2[:,2],color='yellow')

plt.legend(['target', 'output'])
plt.show()






