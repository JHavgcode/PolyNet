import numpy as np
import pandas as pd
import keras
import csv
import random as ran
import math
import cmath
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

#%%
#FUNCTIONS FOR THE DATA

#writing a function to calculate quadratic formula
def quadfill(a,b,c):
  target = np.empty([1,4])
  d = (b**2) - (4*a*c)
  sol1 = (-b+cmath.sqrt(d))/(2*a)
  sol2 = (-b-cmath.sqrt(d))/(2*a)
  target[0,0] = sol1.real
  target[0,1] = sol1.imag
  target[0,2] = sol2.real
  target[0,3] = sol2.imag
  return target

#normalizes input coefficients according to largest coefficient
def normalize(arr):
  norm = np.empty([len(arr),3])
  for c in range(0,len(arr)):
    a = max(abs(arr[c,0]),abs(arr[c,1]),abs(arr[c,2]))
    norm[c,0] = arr[c,0]/a
    norm[c,1] = arr[c,1]/a
    norm[c,2] = arr[c,2]/a
  return norm

#writing a function to combine the two network outputs into one array
def squish(arr1,arr2):
  combined = np.empty([len(arr1),4])
  for u in range(0,len(arr1)):
    combined[u,0] = arr1[u,0]
    combined[u,2] = arr1[u,1]
    combined[u,1] = arr2[u,0]
    combined[u,3] = arr2[u,1]
  return combined

#writing a function to compare a number of random input values to their network outputs
def compareper(net_in, net_out,num,savepath):
  dif = np.empty([1,4])
  real = np.empty([1,4])
  dat = np.empty([num,4])
  for x in range(0,num):
    #compare real values to network output values
    r = math.floor(test_size*(len(net_in))*ran.random())
    print(x," Network Input: \t\t",net_in[r,:])
    real = quadfill(net_in[r,0],net_in[r,1],net_in[r,2])
    print(x," Real answers: \t\t", real[0,:])
    print(x," Network output: \t\t", net_out[r,:])
    dif[0,0] = (abs(abs(real[0,0])-abs(net_out[r,0]))/abs(real[0,0]))*100
    dif[0,1] = (abs(abs(real[0,1])-abs(net_out[r,1]))/abs(real[0,1]))*100
    dif[0,2] = (abs(abs(real[0,2])-abs(net_out[r,2]))/abs(real[0,2]))*100
    dif[0,3] = (abs(abs(real[0,3])-abs(net_out[r,3]))/abs(real[0,3]))*100
    print(x," Absolute Difference: \t", dif[0,:],"\n")
    dat[x,:] = dif
  df = pd.DataFrame(
      data = {'real1':dat[:,0],'imag1':dat[:,1],'real2':dat[:,2],'imag2':dat[:,3]}
  )
  df.to_csv(savepath + '_per.csv', index=False)
  return

#calculates the absolute difference between two arrays and saves the results to a csv at the filepath
def compareabs(net_in, net_out,num):
  dif = np.empty([1,4])
  real = np.empty([1,4])
  avg = np.empty([1,4])
  dat = np.empty([num,4])
  for x in range(0,num):
    #compare real values to network output values
    r = math.floor(test_size*(len(net_in))*ran.random())
    #print("Test Number:",x)
    #print("Network Input: \t\t\t",net_in[r,:])
    real = quadfill(net_in[r,0],net_in[r,1],net_in[r,2])
    #print("Real answers: \t\t\t", real[0,:])
    #print("Network output: \t\t", net_out[r,:])
    dif[0,0] = (abs(abs(real[0,0])-abs(net_out[r,0])))
    dif[0,1] = (abs(abs(real[0,1])-abs(net_out[r,1])))
    dif[0,2] = (abs(abs(real[0,2])-abs(net_out[r,2])))
    dif[0,3] = (abs(abs(real[0,3])-abs(net_out[r,3])))
    #print("Absolute Difference: \t\t", dif[0,:],"\n")
    #if(dif[0,0] > 10):
      #print("\n",x,"\n")

    dat[x,:] = dif
    
  avg = np.mean(dat, axis = 0)
  #print(avg)
  
  df = pd.DataFrame(
      data = {'real1':dat[:,0],'imag1':dat[:,1],'real2':dat[:,2],'imag2':dat[:,3]}
  )
  df.to_csv('data.csv', index=False)
  return avg
#%%
  #BUILDING DATASET
#fix random seed for reproducibility
np.random.seed(123)
#variable for size of data arrays
size = 200000

#variable for range of input values
Range = 10000


#creates empty array for the different coefficients c1x^2 + c2x + c3
#should look like this: c1 c2 c3
c = np.empty([size,3],dtype='double')

#Creates array for answers to the inputs
#should look like this: real(root 1) imag(root 1) real(root 2) imag(root 2) 
a = np.empty([size,4],dtype='double')

#Creates the input data and fills in the arrays
for x in range(0,size):
  for y in range(0,3):
    c[x,y] = (ran.randrange(-Range,Range))
    if(c[x,0] == 0):
      c[x,0] = 1

#calculates the answers and handles the imaginary numbers, setting a1i or a2i to 1 when it's imaginary
#uses try catch type statement
for x in range(0,size):
  a[x,:] = quadfill(c[x,0],c[x,1],c[x,2])

#norm = normalize(c)

#Normalizing the data
sc = StandardScaler()
norm = sc.fit_transform(c)

test_size = 0.2

#creates sets of input and output data to train and test the network on
c_train,c_test,a_train,a_test = train_test_split(norm,a,test_size = 0.2)
#%%
#BUILDING THE NETWORK

#Makes a sequential model where each layer gets added in series
model = Sequential()
#Creates an input layer with 3 inputs  that each connect to 9 nodes
model.add(Dense(9, input_dim=3,activation = 'linear'))
#Creates a hidden layer with 12 nodes
model.add(Dense(12,activation = 'tanh'))
model.add(Dense(16,activation = 'tanh'))
model.add(Dense(20,activation = 'tanh'))
model.add(Dense(24,activation = 'tanh'))
model.add(Dense(4,activation = 'linear'))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
#%%
#TRAINING THE NETWORK

history = model.fit(c_train, a_train,validation_data = (c_test,a_test), epochs=10, batch_size=100)
#%%
#TESTING NETWORK AND SAVING RESULTS

model_pred = model.predict(c_test)
aver = compareabs(c_test,model_pred,100)

objects = ('Real 1','Imaginary 1','Real 2','Imaginary 2')
y_pos = np.arange(len(objects))
performance = aver

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Absolute Difference')
plt.title('Average Absolute Difference of Roots')

plt.show()


model.save('Polynet_example')
#%%
#LOADING MODEL
#model = keras.models.load_model('Polynet_example')
#model_pred = model.predict(c_test)
#aver = compareabs(c_test,model_pred,10)