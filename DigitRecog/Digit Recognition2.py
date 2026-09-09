import time
import json
import numpy as np
from sklearn.datasets import load_digits

start = time.time()
digits = load_digits()

X = digits.data
Y = digits.target

alpha = 0.01

X_normalized = X/16.0

# print(Y.shape)

OneHotEncode = np.zeros((Y.shape[0],10))

def OneHotEncodeArray(Y,OneHotEncode):
    for i in range(Y.shape[0]):
        col = Y[i]
        OneHotEncode[i][col] = 1
    return OneHotEncode


x = np.zeros((X.shape[0],X.shape[1],1))

def Reshape_x(X_normalized,x):
    for i in range(X.shape[0]):
        x[i] = X_normalized[i].reshape(64,1)
    return x

Reshaped_OHE = np.zeros((Y.shape[0],10,1))

def Reshape_y(OneHotEncode,Reshaped_OHE):
    for i in range(Y.shape[0]):
        Reshaped_OHE[i] = OneHotEncode[i].reshape(10,1)
    return Reshaped_OHE

def Relu(x):
    return np.maximum(x,0)

def Softmax(x):
    x_shifted = x-np.max(x)
    return np.exp(x_shifted) / np.sum(np.exp(x_shifted))

def Relu_Derivative(x):
    return (x>0).astype(float)

def CrossEntropyLoss(Reshaped_OHE,a3):
    return -np.sum(Reshaped_OHE*np.log(a3+1e-15))






W1 = np.random.randn(32,64)
W2 = np.random.randn(16,32)
W3 = np.random.randn(10,16)
b1 = np.zeros((32,1))
b2 = np.zeros((16,1))
b3 = np.zeros((10,1))

OneHotEncode = OneHotEncodeArray(Y,OneHotEncode)
Reshaped_OHE = Reshape_y(OneHotEncode,Reshaped_OHE)
x = Reshape_x(X_normalized,x)

# print(Y[:20])
# print(Y[-20:])
# train_size = int(round(0.8*X.shape[0]))
# print("Train label counts:", np.bincount(Y[:train_size]))
# print("Test label counts:", np.bincount(Y[train_size:]))

for i in range(10000):
    # start = time.time()
    for j in range(int(round(0.8*X.shape[0]))): 
        z1 = W1@x[j]+b1 #Shape = 32*1
        a1 = Relu(z1) #Shape = 32*1

        z2 = W2@a1+b2 #Shape = 16*1
        a2 = Relu(z2) #Shape = 16*1

        z3 = W3@a2+b3 #Shape = 10*1
        a3 = Softmax(z3) #Shape = 10*1

        Loss = CrossEntropyLoss(Reshaped_OHE[j],a3)
        # print(f"Loss :- {Loss}")


        dz3 = a3-Reshaped_OHE[j] #Shape = 10*1

        dW3 = dz3@a2.T
        db3 = dz3

        dz2 = (W3.T@dz3) * Relu_Derivative(z2)
        dW2 = dz2@a1.T
        db2 = dz2

        dz1 = (W2.T@dz2) * Relu_Derivative(z1)
        dW1 = dz1@x[j].T
        db1 = dz1

        W3 -= alpha*dW3
        b3 -= alpha*db3

        W2 -= alpha*dW2
        b2 -= alpha*db2

        W1 -= alpha*dW1
        b1 -= alpha*db1
    # end = time.time()
    # print(f"One epoch took {end - start:.2f} seconds")
    # print(f"Estimated 10000 epochs: {(end - start) * 10000 / 60:.1f} minutes")

correct = 0
for k in range(int(round((X.shape[0]-0.8*X.shape[0])))):
    z1 = W1@x[k+int(round(0.8*X.shape[0]))] + b1
    a1 = Relu(z1)

    z2 = W2@a1 + b2
    a2 = Relu(z2)

    z3 = W3@a2 + b3
    a3 = Softmax(z3)

    Prediction = np.argmax(a3)

    if Prediction==Y[k+int(round(0.8*X.shape[0]))] :
        correct += 1

accuracy = correct / int(round((X.shape[0]-0.8*X.shape[0])))
print(f"Test accuracy: {accuracy * 100:.2f}%")



weights = {
    "W1": W1.tolist(), "b1": b1.tolist(),
    "W2": W2.tolist(), "b2": b2.tolist(),
    "W3": W3.tolist(), "b3": b3.tolist()
}

with open("trained_weights.json", "w") as f:
    json.dump(weights, f)

print("Weights saved!")