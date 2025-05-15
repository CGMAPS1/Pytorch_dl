import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import pandas as pd


#creating class:
class Myclass(nn.Module):#inherting base class nn.Moduke
    #Input features(4 features of the flower)
    #Hidden layer 1
    #H2(n)
    #output (3 classes of iris flower)
    def __init__(self,in_features=4,h1=8,h2=9,out_features=3):# default constructor
                
        super().__init__()#inherited class constructor calling 
        
        self.fc1=nn.Linear(in_features,h1)
        self.fc2=nn.Linear(h1,h2)
        self.out=nn.Linear(h2,out_features)
        
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.out(x)
        return x

#for randomisation
torch.manual_seed(7)
#create model object
model=Myclass()

#Loading the dataset
df=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
df['species'] = df['species'].str.replace("Iris-", "", regex=False)
print(df.head())
print(df.shape)
#print(df.columns)

#Changing the dataset according to the model requirements
#M.1
'''#changing the last column to float for multi-class classification
df['species']=df['species'].replace({
  'setosa':0.0,
  'versicolor':1.0,
  'virginica':2.0   
})'''
#M.2->
label={
    'setosa':0.0,
    'versicolor':1.0,
    'virginica':2.0   
}
df['species']=df['species'].map(label).astype(float)

X=df.drop(columns='species')
y=df['species']

#convert to numpy arrays
X=X.values
y=y.values
#print(y)

#importing scikit learn to split the train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=7)

#converting numpu arrays to torch.tensors
# convert numpy arrays to torch tensors with correct types
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()

print(y_test)

#Loss calculation(criterion of model to measure the error)
criterion=nn.CrossEntropyLoss()
#choose optimizer and lr=learning rate()
optim=torch.optim.Adam(model.parameters(),lr=0.01)

print(model.parameters)

#Training our models
epochs=100
losses=[]

for i in range(epochs):
  #Go forward and get the prediction
  y_pred=model.forward(X_train)#get predicted data
  
  
  #measure the loss
  loss=criterion(y_pred,y_train)
  
  #keep track
  losses.append(loss.item())
  
  # print loss in each epoch
  print(f'Epoch: {i+ 1}, Loss: {loss}')
  
  
  #backpropogation
  optim.zero_grad()
  loss.backward()
  optim.step()

# EVALUATE MODEL ACCURACY
with torch.no_grad():  # No gradient calculation needed
    y_pred_test = model(X_test)  # Forward pass on test data
    predicted = torch.argmax(y_pred_test, dim=1)  # Get index of highest logit (class prediction)
    correct = (predicted == y_test).sum().item()  # Count how many are correct
    total = y_test.size(0)  # Total test samples
    accuracy = 100 * correct / total  # Accuracy in percentage
print(f'\nTest Accuracy: {accuracy:.2f}%')# Test Accuracy: 96.67% found 😍😍😍😍🤞🤞🤞🤞🤞🤞


#Plotting graph of loss formore detail
plt.plot (range(epochs),losses)
plt.ylabel("Loss/error")
plt.xlabel('Epoch')
plt.legend
plt.show()
