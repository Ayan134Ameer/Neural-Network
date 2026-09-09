import numpy as np

X = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],
              [1,0,0],[1,0,1],[1,1,0],[1,1,1]])
y = np.array([0,1,1,0,1,0,0,1])

W1 = np.random.randn(4,3)   # layer 1: 3 inputs -> 4 neurons
b1 = np.zeros((4,1))

W2 = np.random.randn(3,4)   # layer 2: 4 inputs -> 3 neurons
b2 = np.zeros((3,1))

W3 = np.random.randn(1,3)   # layer 3: 3 inputs -> 1 output
b3 = np.zeros((1,1))

alpha = 0.5

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return x*(1-x)


for i in range(10000):
    for j in range(8):
        x = X[j].reshape(3,1)
        y_true = y[j]

        z1 = W1@x + b1 #Shape = 4*1
        a1 = sigmoid(z1) #Shape = 4*1

        z2 = W2@a1 + b2 #Shape = 3*1
        a2 = sigmoid(z2) #Shape = 3*1

        z3 = W3@a2 + b3 #Shape = 1*1
        a3 = sigmoid(z3) #Shape = 1*1
        Cost = (y_true-a3)**2

        d_z3 = -2*(y_true-a3) * sigmoid_deriv(a3) #Shape = 1*1
        dW3 =  (d_z3@a2.T)    #Shape = 1*3
        db3 = d_z3 #Shape = 1*1

        d_z2 = (W3.T@d_z3) * sigmoid_deriv(a2)  #Shape = 3*1
        dW2 = (d_z2@a1.T) #Shape = 3*4
        db2 = d_z2

        d_z1 = (W2.T@d_z2) * sigmoid_deriv(a1) #Shape = 4*1
        dW1 = (d_z1@x.T) #Shape = 4*3
        db1 = d_z1

        W3 -= alpha*dW3
        b3 -= alpha*db3

        W2 -= alpha*dW2
        b2 -= alpha*db2

        W1 -= alpha*dW1
        b1 -= alpha*db1



for k in range(8):
    input = X[k].reshape(3,1)
    y_true = y[k]
    z1 = W1@input + b1 #Shape = 4*1
    a1 = sigmoid(z1) #Shape = 4*1

    z2 = W2@a1 + b2 #Shape = 3*1
    a2 = sigmoid(z2) #Shape = 3*1

    z3 = W3@a2 + b3 #Shape = 1*1
    a3 = sigmoid(z3) #Shape = 1*1

    print(f"{X[k]} || {y_true} || Predicted :- {a3}")


