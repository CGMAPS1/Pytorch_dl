import torch 

# Create a 2x3 tensor with random integers between 0 and 9, cast to float
e = torch.randint(size=(2, 3), low=0, high=10).to(dtype=torch.float32)
print("Tensor e:\n", e)

# Sum of all elements
print("\nSum of all elements:")
print(torch.sum(e))

# Sum along columns (dim=0)
print("\nSum along dim=0 (column-wise sum):")
print(e.sum(dim=0))  # No need to wrap this in torch()

# Mean of all elements
print("\nMean of all elements:")
print(torch.mean(e))

# Median of all elements (flattened)
print("\nMedian of all elements:")
print(torch.median(e))


#Matrix operations in tensor
x=torch.randint(0,10,size=(3,3),dtype=torch.float32)
y=torch.randint(0,100,size=(3,3),dtype=torch.float32)

#1)Matrix Multiply
print(torch.matmul(x,y))
#2)Matrix dot product
print(torch.dot(x[1],y[1]))#applicable for only 1d vectors
#finding the transpose of a matrix 
print(torch.transpose(x,0,1))

#finding the determinant 
print(torch.linalg.det(y))
print(torch.inverse(y))

#some special functions
z=torch.randint(10,20,size=(3,3),dtype=torch.float32)
#exp
print(torch.exp(z))
#log
print(torch.log(z))
#sine function
print(torch.sin(z))
#sigmoid function
print(torch.sigmoid(z))
#softmax function
print(torch.softmax(z,dim=0))


#Implace operations
m=torch.rand(2,3)
n=torch.rand(2,3)
# for amking permanent changes in any tensor always use undescore symbol
print(m.add_(n))
print(m)

print(m.relu_())
print(m)


#SHALLOW COPY AND DEEP copy in python :a=b(shallow copy) b=a.clone()
m=n.clone()
print(m)
print(n)

#Tensor operations in gpu

#to check if gpu is available or not
import torch 
print(torch.cuda.is_available())
device=torch.device('cuda')
#creating a tensor on gpu
print(torch.rand((2,3),device=device))
a=torch.rand(2,3)
torch.manual_seed(10)
b=torch.linspace(1,10,9).reshape((3,3))

c=a.to(device)
print(b)
print(c)

#Reshaping the tensors
import torch
p=torch.ones(4,4)
print(p)
q=p.reshape(2,4,2)
print(q)

#using permute (permutes the rows ,columns and height of the torch object)
torch.manual_seed(10)
b=torch.randn(2,3,4)
print(b)
print(b.permute(2,1,0))

#reshaping to 1d by using flatten fucntion
print(c.flatten())

#Unsqeezing operations in torch
import torch
a = torch.tensor([1, 2, 3])
print("Original:", a.shape)  # torch.Size([3])

b = torch.unsqueeze(a, dim=0)  # Add a new outer dimension
print("After unsqueeze(0):", b.shape)
print(b)  # torch.Size([1, 3])-> 1 row and three column

c = torch.unsqueeze(a, dim=1)  # Add a new inner dimension
print("After unsqueeze(1):", c.shape)  # torch.Size([3, 1])->3 row and 
print(c)

#squeeze operations in torch
a = torch.tensor([[1.0, 2.0, 3.0]])  # Shape: [1, 3]->2d matrix
b = torch.tensor([1.0, 2.0, 3.0])    # Shape: [3]->1d matrix

# They can't be added unless we match shape
a_squeezed = a.squeeze()  # Now shape is [3]
print("a + b:", a_squeezed + b)

#Numpy to torch use-> torch.from_numpy(array)
#vice versa use -> torch.tensor.numpy()
#eg->1
import numpy as np
import torch

np_array = np.array([1, 2, 3])
torch_tensor = torch.from_numpy(np_array)

print(torch_tensor)        # tensor([1, 2, 3])
print(torch_tensor.dtype)  # dtype is inferred (usually int64 or float64)

#eg-2
torch_tensor = torch.tensor([1, 2, 3])
np_array = torch_tensor.numpy()  # ✅ No need to call torch.tensor.numpy()

print(np_array)        # [1 2 3]
print(type(np_array))  # <class 'numpy.ndarray'>


#:))) session is over 