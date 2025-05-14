import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
#loading the dataset 
df=pd.read_csv('https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv')
#print(df.tail())
print(df.shape)
#dropping the unwanted columns 
df.drop(columns=['id', 'Unnamed: 32'], inplace=True, errors='ignore')
print(df.head())#print to check whether the changes are made or not 
print(df.columns.tolist())

#Train and test data split 
X_train,X_test,y_train,y_test=train_test_split(df.iloc[:,1:],df.iloc[:,0],test_size=0.2)


#scaling the data
scalar=StandardScaler()
X_train=scalar.fit_transform(X_train)
X_test=scalar.transform(X_test)

print(X_train)
print(y_train)

#Label encoding 
encoder=LabelEncoder()
y_train=encoder.fit_transform(y_train)
y_test=encoder.transform(y_test)
print(y_train)
print(y_test)

#Converting numpy  arrays to torch tensors
X_train_tensor=torch.from_numpy(X_train)
X_test_tensor=torch.from_numpy(X_test)
y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)


print(y_train_tensor)
print(X_train_tensor)

#defining the model
class myfirstPro():
    def __init__(self,X):
        self.weight=torch.rand(X.shape[1],1,dtype=torch.float64,requires_grad=True)
        self.bais=torch.rand(1,dtype=torch.float64,requires_grad=True)
        
    #forward propogation
    def forward(self,X):
        z=torch.matmul(X,self.weight)+self.bais
        y_pred=torch.sigmoid(z)
        return y_pred
    
    #calculation of loss 
    def loss(self,y_pred,y_train_tensor):
        #clamp predictions to avoid log(0) as many are 0
        epsilon=1e-8
        y_pred=torch.clamp(y_pred,epsilon,1-epsilon)
        
        #calculate loss
        loss=-(y_train_tensor*torch.log(y_pred)+(1-y_train_tensor)*torch.log(1-y_pred))
        return loss.mean()

learning_rate=0.01
epochs=100
#Training pipeline 
#create model=
model=myfirstPro(X_train_tensor)

#define for loop epochs
for i in range(epochs):
    #forward pass
    y_pred=model.forward(X_train_tensor)
    #loss cal
    loss=model.loss(y_pred,y_train_tensor)
    # Compute gradients without accumulation
    grads = torch.autograd.grad(loss, [model.weight, model.bais])

    # grads is a tuple: (grad_weights, grad_bias)
    grad_weight, grad_bais = grads

    # Manual parameter update
    with torch.no_grad():
        model.weight -= learning_rate * grad_weight
        model.bais -= learning_rate * grad_bais
        
    # print loss in each epoch
    print(f'Epoch: {i+ 1}, Loss: {loss.item()}')


print(model.bais)

#Model Evaluation
with torch.no_grad():
    y_pred=model.forward(X_test_tensor)
    y_pred=(y_pred>0.5).float()
    accuracy=(y_pred==y_test_tensor).float().mean()
    print(f'Accuracy :{accuracy.item():.4f}')