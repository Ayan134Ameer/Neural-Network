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
        x = X[j].reshape(2,1)
        y_true = y[j]
        z_1 = W1@x + b1 # Linear combinations for the first layer
        a_1 = sigmoid(z_1) # Activations of linear combinations
        z_2 = W2@a_1+b2
        a_2 = sigmoid(z_2)
        Cost = (y_true-a_2)**2


        M1 = -2*(y_true-a_2[0][0])*(sigmoid_deriv(a_2[0][0]))
        derivAct_L2_weightArray = M1*a_1
        derivAct_L2_weightArray = derivAct_L2_weightArray.reshape(1,4)

        # W2 = W2.reshape(4,1)


        M2 = M1*W2[0][0]*(sigmoid_deriv(a_1[0][0]))
        derivAct1_L1_weightArray = M2*x
        derivAct1_L1_weightArray = derivAct1_L1_weightArray.reshape(1,2)


        M3 = M1*W2[0][1]*(sigmoid_deriv(a_1[1][0]))
        derivAct2_L1_weightArray = M3*x
        derivAct2_L1_weightArray = derivAct2_L1_weightArray.reshape(1,2)



        M4 = M1*W2[0][2]*(sigmoid_deriv(a_1[2][0]))
        derivAct3_L1_weightArray = M4*x
        derivAct3_L1_weightArray = derivAct3_L1_weightArray.reshape(1,2)



        M5 = M1*W2[0][3]*(sigmoid_deriv(a_1[3][0]))
        derivAct4_L1_weightArray = M5*x
        derivAct4_L1_weightArray = derivAct4_L1_weightArray.reshape(1,2)

        W2 -= lr*derivAct_L2_weightArray
        b2 -= lr*M1
        W1[0] -= lr*derivAct1_L1_weightArray.flatten()
        b1[0][0] -= lr*M2
        W1[1] -= lr*derivAct2_L1_weightArray.flatten()
        b1[1][0] -= lr*M3
        W1[2] -= lr*derivAct3_L1_weightArray.flatten()
        b1[2][0] -= lr*M4
        W1[3] -= lr*derivAct4_L1_weightArray.flatten()
        b1[3][0] -= lr*M5

x_out = X[3].reshape(2,1)
y_true1 = y[3]
z_11 = W1@x_out + b1 # Linear combinations for the first layer
a_11 = sigmoid(z_11) # Activations of linear combinations
z_22 = W2@a_11+b2
a_22 = sigmoid(z_22)

print(a_22)