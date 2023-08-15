#!/usr/bin/python3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt




def uni_var_linear_reg(X, y, max_its, eta):
    # logistic_reg learn logistic regression model using gradient descent
    # Inputs:
    #        X : one feature of X_train
    #        y : y_train
    #        max_its: maximum number of iterations to run for
    #        eta: learning rate
    # Outputs:
    #        m : m after training
    #        b : b after training
    #        Variance_explained : the variance explained value
    t = 0
    m = 0
    b = 0
    while(t<max_its):
        hy = m * X + b
        m_gd = np.sum((-2) * X * (y - hy))/len(X)
        b_gd = np.sum((-2) * (y - hy))/len(X)
        m = m - eta * m_gd
        b = b - eta * b_gd
        t = t + 1
        
        # if t%10000 == 0:
        #     print(np.mean((y - hy)**2))
    loss_mse = np.mean((y - hy)**2)
    Variance_explained = 1 - loss_mse/np.var(y)
    return m, b, Variance_explained

def multi_var_linear_reg(X, y, max_its, eta):
    # logistic_reg learn logistic regression model using gradient descent
    # Inputs:
    #        X : one feature of X_train
    #        y : y_train
    #        max_its: maximum number of iterations to run for
    #        eta: learning rate
    # Outputs:
    #        m : m after training
    #        Variance_explained : the variance explained value
    t = 0
    m = np.zeros(X.shape[1])
    while(t<max_its):
        hy = np.dot(X, m) 
        m_gd = np.sum((-2) * X * (y - hy)[:, None],axis=0)/len(y)
        m = m - eta * m_gd
        t = t + 1
        
        # if t%10000 == 0:
        #     print(np.mean((y - hy)**2))
    loss_mse = np.mean((y - hy)**2)
    Variance_explained = 1 - loss_mse/np.var(y)
    return m, Variance_explained



##normalize data
def normalization(x, col):
    mean = np.zeros(col)
    var = np.zeros(col)
    for i in range(col):
        mean[i] = np.mean(x[:,i])
        var[i] = np.std(x[:,i])
    # mean = np.mean(x, axis=0)
    # var = np.std(x, axis=0)
    return mean,var

def main():
    ## Load data
    data = pd.read_excel("Concrete_Data.xls", "Sheet1")
    data = data.values
    num_col_train=data.shape[1]
    num_row_train=data.shape[0]
    x = data[:,:-1]
    y = data[:,-1]
    X_train = np.zeros([900,num_col_train-1])
    y_train = np.zeros([900,1])
    X_test = np.zeros([130,num_col_train-1])
    y_test = np.zeros([130,1])
    X_train,X_test,y_train,y_test = train_test_split(x, y, train_size = 900, test_size = 130)
    
    # ##  Uni-variate
    # eta_arr = [10**(-5),10**(-5)*5,10**(-4),10**(-6)*5,10**(-4),10**(-6),10**(-6),10**(-4)]
    # for i in range(X_train.shape[1]):
    #     x_feature = X_train[:,i]
    #     x_feature_test = X_test[:,i]
    #     print("feature:      " +str(i))
    #     m, b, Variance_explained = uni_var_linear_reg(x_feature, y_train, 100000, eta_arr[i])
    #     print ("m:    "+str(m))
    #     print ("b:    "+str(b))
    #     print ("uni_Variance_explained_train:    " + str(Variance_explained))
    #     hy_test = m * x_feature_test + b
    #     loss_mse_test = np.mean((y_test - hy_test)**2)
    #     Variance_explained_test = 1 - loss_mse_test/np.var(y_test)
    #     print ("uni_Variance_explained_test:    " + str(Variance_explained_test) + "\n")
    #     plt.scatter(x_feature, y_train)
    #     plt.plot(x_feature, m * x_feature + b, color='red')
    #     plt.grid()
    #     plt.xlabel('Feature')
    #     plt.ylabel('Concrete Compressive Strength')
    #     plt.show()



    ##  Uni-variate-nor
    eta_arr = [10**(-4),10**(-4),10**(-4),10**(-4),10**(-4),10**(-4),10**(-4),10**(-4)]
    mean, var = normalization(X_train, X_train.shape[1])
    X_train_nor = (X_train-mean)/var
    X_test_nor = (X_test-mean)/var
    plt.hist(X_train, bins=5)    ##  show image of X and X_nor
    plt.show()
    plt.hist(X_train_nor, bins=5)
    plt.show()
    for i in range(X_train_nor.shape[1]):
        x_feature = X_train_nor[:,i]
        x_feature_test = X_test_nor[:,i]
        print("feature:      " +str(i))
        m, b, Variance_explained = uni_var_linear_reg(x_feature, y_train, 100000, eta_arr[i])
        print ("m:    "+str(m))
        print ("b:    "+str(b))
        print ("uni_Variance_explained_train:    " + str(Variance_explained))
        hy_test = m * x_feature_test + b
        loss_mse_test = np.mean((y_test - hy_test)**2)
        Variance_explained_test = 1 - loss_mse_test/np.var(y_test)
        print ("uni_Variance_explained_test:    " + str(Variance_explained_test) + "\n")
        plt.scatter(x_feature, y_train)
        plt.plot(x_feature, m * x_feature + b, color='red')
        plt.grid()
        plt.xlabel('Feature')
        plt.ylabel('Concrete Compressive Strength')
        plt.show()
    
    ##   Multi-variate
    X_train_multi = np.column_stack((np.ones(900), X_train))
    X_test_multi = np.column_stack((np.ones(130), X_test))
    m_multi, Variance_explained_multi = multi_var_linear_reg(X_train_multi, y_train, 100000, 10**(-7))
    print ("m:   " + str(m_multi[1:len(m_multi)]))
    print ("b:   " + str(m_multi[0]))
    print ("multi_Variance_explained_train:    " + str(Variance_explained_multi))
    hy_test_multi = np.dot(X_test_multi, m_multi)
    loss_mse_test_multi = np.mean((y_test - hy_test_multi)**2)
    Variance_explained_test_multi = 1 - loss_mse_test_multi/np.var(y_test)
    print ("milti_Variance_explained_test:    " + str(Variance_explained_test_multi) + "\n")

    ##  Multi-variate_nor
    mean, var = normalization(X_train, X_train.shape[1])
    X_train_multi_nor = (X_train-mean)/var
    X_test_multi_nor = (X_test-mean)/var
    X_train_multi_nor = np.column_stack((np.ones(900), X_train_multi_nor))
    X_test_multi_nor = np.column_stack((np.ones(130), X_test_multi_nor))
    m_multi_nor, Variance_explained_multi_nor = multi_var_linear_reg(X_train_multi_nor, y_train, 100000, 10**(-3))
    print ("m:   " + str(m_multi_nor[1:len(m_multi_nor)]))
    print ("b:   " + str(m_multi_nor[0]))
    print ("multi_Variance_explained_train_nor:    " + str(Variance_explained_multi_nor))
    hy_test_multi_nor = np.dot(X_test_multi_nor, m_multi_nor)
    loss_mse_test_multi_nor = np.mean((y_test - hy_test_multi_nor)**2)
    Variance_explained_test_multi_nor = 1 - loss_mse_test_multi_nor/np.var(y_test)
    print ("multi_Variance_explained_test_nor:    " + str(Variance_explained_test_multi_nor) + "\n")
    

    
if __name__ == "__main__":
    main()
