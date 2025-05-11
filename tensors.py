import torch as t
import matplotlib.pyplot as plt
import pandas as pd

print(t.__version__)

# creating tensors

# using empty 
s = t.empty(2, 3)
print(s)

# check type
print(type(s))

# using zeros
s1 = t.zeros(2, 3)
print(s1)

# using ones
s2 = t.ones(3, 4)
print(s2)

# using rand -> for printing the random values 
s3 = t.rand(2, 3)
print(s3)

# seeding torch's random generator
t.manual_seed(10)
s4 = t.randn(3, 4)
print(s4)

# another manual seed
t.manual_seed(110)
s5 = t.randn(3, 5)
print(s5)

# torch.full(size, val): fill tensor with specific value
print("\n1. torch.full:")
a = t.full((2, 3), 7)
print(a)  # 2x3 tensor filled with 7s

# torch.arange(start, end): like Python's range
print("\n2. torch.arange:")
b = t.arange(1, 10)  # 1 to 9
print(b)

# torch.linspace(start, end, steps): linearly spaced values
print("\n3. torch.linspace:")
c = t.linspace(0, 1, steps=5)  # 5 values from 0 to 1
print(c)

# torch.eye(n): identity matrix
print("\n4. torch.eye:")
d = t.eye(4)
print(d)

# torch.randint(low, high, size): random integers between low and high
print("\n5. torch.randint:")
e = t.randint(0, 100, (3, 4))  # 3x4 tensor with values from 0 to 99
print(e)

#manually creating arrays by yourselves

s6=t.tensor([[1,2],[4,5]])
print(s6)

import torch

torch.manual_seed(3)
print(torch.rand(2))

torch.manual_seed(42)
print(torch.rand(2))

torch.manual_seed(3)
print(torch.rand(2))  # Same as first print


#uSING SHAPE OF A TENSOR
print(s1.shape)

#empty_like function used to create a tensor of same shape 
print(torch.empty_like(s2))

#zeros_like(x) and ones_like(x) x is the tensor passed
print(torch.zeros_like(s3))
print(torch.ones_like(s4))

#tensor datatypes
print(s6.dtype)


#assign datatyoe
print(torch.tensor([[1,2,4]],dtype=torch.float64))

#changing the dataype
print(s1.to(torch.int32))

print(torch.rand_like(s4,dtype=torch.float32))


#Mathematical operations in tensor
print(s2+2)
print(s3/3)

print((s4*100)//3)#integer divison

#Elementwise operations in tensor(size of  both the tensor objects should be same )


m = t.arange(1, 10)         # 1D tensor: [1, 2, ..., 9]
m2d = m.reshape(3, 3)       # reshape to 3x3

n=t.arange(2,11).reshape(3,3)
print(m2d+n)
print(m2d/n)

#moreover operations like floor,ceil,round,abs,clamp can also be used
print(n.sum(dim=0))  # sum along columns
print(m2d.sum(dim=1))  # sum along rows



#also operations like mean ,mediam ,mode,sd can also be found

#using argmax,argmin
print(t.argmax(n))#tells the position lof largest element 
print(t.argmin(m2d))#tell the postion of smallest element 




