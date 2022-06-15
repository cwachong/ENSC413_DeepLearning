import numpy as np
#scipy.special for the sigmoid function expit()
import scipy.special    
#to measure elapsed time
from timeit import default_timer as timer 
#pandas for reading CSV files
import pandas as pd 
#Keras
import keras 
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.utils import np_utils 
#support tensorflow warnings 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#timer for computation time
start_t = timer()

#number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
 
#Learning rate
learning_rate = 0.001

#Create keras model
model = Sequential()
model.add(Dense(hidden_nodes, activation='sigmoid', input_shape=(input_nodes,), bias=False))
model.add(Dense(output_nodes,activation='sigmoid', bias=False))

#Print model summary
model.summary()

#set optimizer (Adam derived from Stoch. Grad. Desc.)
opt = optimizers.Adam(lr=learning_rate)

#Define error criterion ("loss"), the optimizer and an optional metric to monitor during training
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#load the mnist training data CSV file into a list
df = pd.read_csv("mnist_csv/mnist_test.csv", header=None)
#columns 1-784 are input values
x_train = np.asfarray(df.loc[:, 1:input_nodes].values)
x_train /= 255.0
#column 0 is desired label
labels = df.loc[:,0].values
#convert labels to OHE for ease of training 
y_train = np_utils.to_categorical(labels,output_nodes)

#train the neural network
epochs = 5 
batch_size = 32

#train the model 
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

#test the model 

#load the mnist test data CSV file into a list
test_data_file = open("mnist_csv/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#scorecard for how well the network performs, initially empty
scorecard = []

#go through all the data in the test data set, one by one
for record in test_data_list:
    # split the record by the ',' commas
    data_sample = record.split(',')
    #correct answer is first value
    correct_label = int(data_sample[0])
    #scale and shift the inputs 
    inputs = np.asfarray(data_sample[1:]) / 255.0

    #make prediction
    outputs = model.predict(np.reshape(inputs, (1, len(inputs))))

    #the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    #append correct or incorrect to list
    if (label == correct_label):
        #network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        #network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    pass

#calculate the accuracy, the fraction of correct answers
scorecard_array = np.asarray(scorecard)
print("accuracy = {}".format(scorecard_array.sum()/scorecard_array.size))

#stop the timer
end_t = timer()
print("elapsed time = {} seconds".format(end_t - start_t))

