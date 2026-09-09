import numpy as np
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

np.random.seed(1)

W1 = np.random.randn(4,2)
b1 = np.zeros((4,1))
W2 = np.random.randn(1,4)
b2 = np.zeros((1,1))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(y):
    return y*(1-y)

lr = 0.5

for i in range(10000):
    for j in range(4):
        x = X[j].reshape(2,1) #Shape = 2*1
        y_true = y[j]
        z_1 = W1@x + b1 #Shape = 4*1
        a_1 = sigmoid(z_1) #Shape = 4*1
        z_2 = W2@a_1 + b2 #Shape = 1*1
        a_2 = sigmoid(z_2) #Shape = 1*1
        Cost = (y_true-a_2)**2

        d_z2 = -2*(y_true-a_2) * sigmoid_deriv(a_2) #Shape = 1*1
        dW2 = d_z2 @ a_1.T #Shape = 1*2
        db2 = d_z2 #Shape = 1*1

        d_z1 = (W2.T@d_z2) * sigmoid_deriv(a_1) #Shape = 4*1
        dW1 = d_z1 @ x.T #Shape = 4*2
        db1 = d_z1 #Shape = 1*1

        W2 -= lr*dW2
        b2 -= lr*db2

        W1 -= lr*dW1
        b1 -= lr*db1


for j in range(4):
    input = X[j].reshape(2,1)
    y_true = y[j]

    z1 = W1@input + b1
    a1 = sigmoid(z1)

    z2 = W2@a1 + b2
    a2 = sigmoid(z2)

    print(f"{X[j]} True : {y[j]} || Predicted : {a2} ")



