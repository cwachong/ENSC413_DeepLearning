import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from timeit import default_timer as timer

# read the dataframe from the csv file
df = pd.read_csv("polydata.csv")
# extract x and y values from the corresponding columns in the dataframe 
x = df.loc[:,'x'].values
y = df.loc[:,'y'].values
# now x and y contain the data values from a polynomial

noisy_data = plt.scatter(x, y, c = "c", marker = 'x', label = "Noisy data")          #This is the noisy data
plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatter plot for the noisy data")
plt.legend(handles = [noisy_data])
plt.savefig("Output/Fig0.png")

#Running the code ten times to collect more information for better results
for count in range(0,50):
    #stochastic gradient descent with training-validation split
    #set initial learning rate
    eta = 0.00001
    #set error improvement tolerance 
    epsilon = 0.00001
    # the number of consecutive epochs that validation error is allowed to increase before stopping 
    patience = 15
    # minibatch size
    batch_size = 10

    #Generating an empty array of size "epoch" to store all the epochs for all the runs
    epoch = [0 for y in range(11)]      
    train_errors = []         #training error 2D list
    validate_errors = []      #validation error 2D list
    p_sgd = []      #An array that stores all the lines for each run

    #Running the stochastic gradient descent for each order of the polynomial
    #The loop runs for polynomial orders 1 to 10
    for i in range(0,11):
        #initialize coefficients randomly
        start_t = timer()   #Timer to calculate how much time it takes for the code to find the polynomial
        a = np.random.normal(0, 1, i+1)

        print("\nStochastic gradient descent for order {} with training-validation split:".format(i))
        print("Initial weights")
        print(a)

        if i == 4:
            eta = 0.000005
        elif i == 5:
            eta = 0.000001
        elif i == 6:
            eta = 0.0000003
        elif i == 7:
            eta = 0.000000015
        elif i == 8:
            eta = 0.000000005
        elif i == 9:
            eta = 0.0000000008
        elif i == 10:
            eta = 0.00000000008
        else:
            pass

        #split into train (60%) and validation (40%)
        x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size =0.4, random_state=32)

        #set previous error value to infinity 
        previous_validate_error_value = np.inf
        #compute current error 
        p_current = np.poly1d(a)
        current_validate_error_value = mean_squared_error(p_current(x_validate), y_validate)

        # parameter that measures how many times the validation error increased
        # more precisely, how many times it failed to decrease by more than epsilon 
        error_up = 0

        #Adding a new element to the train and validation errors
        train_errors.append([])
        validate_errors.append([])
        counter = 0
        # iterate until error improvement is less than epsilon and epoch is less than 50000
        while error_up < patience and epoch[i] < 50000:
            epoch[i] += 1 
            if previous_validate_error_value - current_validate_error_value > epsilon:
                error_up = 0
            else:
                error_up += 1

            #randomly select minibatches from the training set and update for each batch 
            indices = np.arange(len(x_train))
            np.random.shuffle(indices)
            for start_idx in range(0, len(x_train) - batch_size + 1, batch_size):
                sample = indices[start_idx : start_idx + batch_size]
                x_batch = x_train[sample]   #Choosing the x values of a batch
                y_batch = y_train[sample]   #Choosing the y values of a batch
                X_batch = np.vander(x_batch, i+1)
                current_error = np.matmul(X_batch, a) - y_batch
                current_gradient = 2.0 * np.matmul(X_batch.transpose(), current_error)
                a -= eta * current_gradient

            previous_validate_error_value = current_validate_error_value
            # compute current errors on training and validation sets
            p_current = np.poly1d(a)
            current_train_error_value = mean_squared_error(p_current(x_train), y_train)
            current_validate_error_value = mean_squared_error(p_current(x_validate), y_validate)

            #add the new errors to the lists
            train_errors[i].append(current_train_error_value)
            validate_errors[i].append(current_validate_error_value)

        print("epoch {}".format(epoch[i]))
        print(a)
        end_t = timer()
        print("It took {}s for the code to find order {} polynomial".format(end_t-start_t, i))
        a_sgd = a

        #define a polynomial based on a_sgd coefficients
        p_sgd.append(np.poly1d(a_sgd))

    #plot
    plt.clf()
    noisy_data = plt.scatter(x, y, c = "c", marker = 'x', label = "Noisy data")          #This is the noisy data
    line_sgd0, = plt.plot(x, p_sgd[0](x), 'r--', label = 'SGD - order {}'.format(0))     #Stochastic gradient descent for order 0
    line_sgd1, = plt.plot(x, p_sgd[1](x), 'b--', label = 'SGD - order {}'.format(1))     #Stochastic gradient descent for order 1
    line_sgd2, = plt.plot(x, p_sgd[2](x), 'k--', label = 'SGD - order {}'.format(2))     #Stochastic gradient descent for order 2
    line_sgd3, = plt.plot(x, p_sgd[3](x), 'y--', label = 'SGD - order {}'.format(3))     #Stochastic gradient descent for order 3
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Plot match for polynomial orders of 0 to 3")
    plt.legend(handles = [noisy_data, line_sgd0, line_sgd1, line_sgd2, line_sgd3])
    plt.savefig("Output/Fig1-{}.png".format(count+1))

    plt.clf()
    noisy_data = plt.scatter(x, y, c = "c", marker = 'x', label = "Noisy data")          #This is the noisy data
    line_sgd4, = plt.plot(x, p_sgd[4](x), 'r--', label = 'SGD - order {}'.format(4))     #Stochastic gradient descent for order 4
    line_sgd5, = plt.plot(x, p_sgd[5](x), 'b--', label = 'SGD - order {}'.format(5))     #Stochastic gradient descent for order 5
    line_sgd6, = plt.plot(x, p_sgd[6](x), 'k--', label = 'SGD - order {}'.format(6))     #Stochastic gradient descent for order 6
    line_sgd7, = plt.plot(x, p_sgd[7](x), 'y--', label = 'SGD - order {}'.format(7))     #Stochastic gradient descent for order 7
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Plot match for polynomial orders of 4 to 7")
    plt.legend(handles = [noisy_data, line_sgd4, line_sgd5, line_sgd6, line_sgd7])
    plt.savefig("Output/Fig2-{}.png".format(count+1))

    plt.clf()
    noisy_data = plt.scatter(x, y, c = "c", marker = 'x', label = "Noisy data")          #This is the noisy data
    line_sgd8, = plt.plot(x, p_sgd[8](x), 'r--', label = 'SGD - order {}'.format(8))     #Stochastic gradient descent for order 8
    line_sgd9, = plt.plot(x, p_sgd[9](x), 'b--', label = 'SGD - order {}'.format(9))     #Stochastic gradient descent for order 9
    line_sgd10, = plt.plot(x, p_sgd[10](x), 'k--', label = 'SGD - order {}'.format(10))     #Stochastic gradient descent for order 10
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Plot match for polynomial orders of 8 to 10")
    plt.legend(handles = [noisy_data, line_sgd8, line_sgd9, line_sgd10])
    plt.savefig("Output/Fig3-{}.png".format(count+1))

    plt.clf()
    noisy_data = plt.scatter(x, y, c = "c", marker = 'x', label = "Noisy data")          #This is the noisy data
    line_sgd4, = plt.plot(x, p_sgd[4](x), 'r--', label = 'SGD - order {}'.format(4))     #Stochastic gradient descent for order 4
    line_sgd6, = plt.plot(x, p_sgd[6](x), 'b--', label = 'SGD - order {}'.format(6))     #Stochastic gradient descent for order 6
    line_sgd8, = plt.plot(x, p_sgd[8](x), 'k--', label = 'SGD - order {}'.format(8))     #Stochastic gradient descent for order 8
    line_sgd10, = plt.plot(x, p_sgd[10](x), 'y--', label = 'SGD - order {}'.format(10))     #Stochastic gradient descent for order 10
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Plot match for polynomial orders of 4, 6, 8, and 10")
    plt.legend(handles = [noisy_data, line_sgd4, line_sgd6, line_sgd8, line_sgd10])
    plt.savefig("Output/Fig4-{}.png".format(count+1))

    plt.clf()
    #plot errors from epoch 10 onwards
    e = range(10, epoch[4])     
    #plotting errors for polynomial order 4
    #line_error_train4, = plt.plot(e, train_errors[4][10:max_epoch], 'b-', label = 'training error order 4')
    line_error_validate4, = plt.plot(e, validate_errors[4][10:], 'g-', label = 'Validation error order 4')
    e = range(10, epoch[5])  
    #plotting errors for polynomial order 5
    #line_error_train5, = plt.plot(e, train_errors[5][10:max_epoch], 'r-', label = 'training error order 5')
    line_error_validate5, = plt.plot(e, validate_errors[5][10:], 'k-', label = 'Validation error order 5')
    e = range(10, epoch[6])  
    #plotting errors for polynomial order 6
    #line_error_train6, = plt.plot(e, train_errors[6][10:max_epoch], 'r-', label = 'training error order 6')
    line_error_validate6, = plt.plot(e, validate_errors[6][10:], 'r-', label = 'Validation error order 6')
    e = range(10, epoch[7])  
    #plotting errors for polynomial order 7
    #line_error_train7, = plt.plot(e, train_errors[7][10:max_epoch], 'r-', label = 'training error order 7')
    line_error_validate7, = plt.plot(e, validate_errors[7][10:], 'y-', label = 'Validation error order 7')
    plt.legend(handles = [line_error_validate4, line_error_validate5, line_error_validate6, line_error_validate7])
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Errors for polynomials of order 4 to 7")
    plt.savefig("Output/Fig5-{}.png".format(count+1))

    plt.clf()
    #plot errors from epoch 5000 onwards
    e = range(10, epoch[4])     
    #plotting errors for polynomial order 4
    #line_error_train4, = plt.plot(e, train_errors[4][10:max_epoch], 'b-', label = 'training error order 4')
    line_error_validate4, = plt.plot(e, validate_errors[4][10:], 'g-', label = 'Validation error order 4')
    e = range(10, epoch[6])  
    #plotting errors for polynomial order 6
    #line_error_train6, = plt.plot(e, train_errors[6][10:max_epoch], 'r-', label = 'training error order 6')
    line_error_validate6, = plt.plot(e, validate_errors[6][10:], 'r-', label = 'Validation error order 6')
    e = range(10, epoch[8])  
    #plotting errors for polynomial order 8
    #line_error_trai86, = plt.plot(e, train_errors[8][50:max_epoch], 'r-', label = 'training error order 8')
    line_error_validate8, = plt.plot(e, validate_errors[8][10:], 'b-', label = 'Validation error order 8')
    e = range(10, epoch[10])  
    #plotting errors for polynomial order 10
    #line_error_train10, = plt.plot(e, train_errors[10][10:max_epoch], 'r-', label = 'training error order 10')
    line_error_validate10, = plt.plot(e, validate_errors[10][10:], 'y-', label = 'Validation error order 10')
    plt.legend(handles = [line_error_validate4, line_error_validate6, line_error_validate8, line_error_validate10])
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Errors for polynomials of order 4, 6, 8, and 10")
    plt.savefig("Output/Fig6-{}.png".format(count+1))

    plt.clf()
    #plot errors from epoch 50 onwards
    e = range(50, epoch[6])  
    #plotting errors for polynomial order 6
    #line_error_train6, = plt.plot(e, train_errors[6][50:max_epoch], 'r-', label = 'training error order 6')
    line_error_validate6, = plt.plot(e, validate_errors[6][50:], 'k-', label = 'Validation error order 6')
    e = range(50, epoch[8])  
    #plotting errors for polynomial order 8
    #line_error_trai86, = plt.plot(e, train_errors[8][50:max_epoch], 'r-', label = 'training error order 8')
    line_error_validate8, = plt.plot(e, validate_errors[8][50:], 'r-', label = 'Validation error order 8')
    plt.legend(handles = [line_error_validate6, line_error_validate8])
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Errors for polynomials of order 6 and 8")
    plt.savefig("Output/Fig7-{}.png".format(count+1))

quit()