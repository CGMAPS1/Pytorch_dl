import torch 
#autograd helps in automatic differentiation for tensor operations 
#enables gradient computation

#Example 1-> y=x^2 calculate dy/dx??

#While making tensors include (requires_grad=True)
x=torch.tensor(3.,requires_grad=True)
print(x)

y=x**2           #x^2
print(y)

y.backward()     #must include 
print(x.grad)    #prints the derivative 

#Example 2-> y=x^2 and z=sin(y) calculate dz/dx ?
#M.1->
x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.retain_grad()  # to allow storing .grad for non-leaf y
z = torch.sin(y)

z.backward()
#y.backward() -> throws error❌❎👨‍🦰 as backward calls aren't allowed else use z.backward(retain_graph=True)
print("dz/dy stored in y.grad (only if retain_grad used):", y.grad)
print("dz/dx stored in x.grad:", x.grad)


#M.2->

# Leaf input variable
x = torch.tensor(2.0, requires_grad=True)

# Forward computation
y = x**2       # y = x²
z = torch.sin(y)  # z = sin(y)

# Derivatives
dy_dx = torch.autograd.grad(y, x, retain_graph=True)[0]  # dy/dx = 2x
dz_dy = torch.autograd.grad(z, y, retain_graph=True)[0]  # dz/dy = cos(y)
dz_dx = torch.autograd.grad(z, x)[0]                     # dz/dx = cos(y) * 2x

# Print all gradients
print(f"dy/dx = {dy_dx.item():.4f}")
print(f"dz/dy = {dz_dy.item():.4f}")
print(f"dz/dx = {dz_dx.item():.4f}")


''' NOTE: 
📊 Summary Table:
Feature	                   tensor.backward()	           torch.autograd.grad()


1)Returns gradients	       ❌ No (stored in .grad)	     ✅ Yes (returns gradients)
2)Requires scalar output   ✅ Yes	                     ❌ No (can handle vector outputs)
3)Manual control	       ❌ No	                         ✅ Yes
4)Higher-order gradients   ❌ Limited	                 ✅ Use create_graph=True
5)Intermediate(non-leaf)   grads❌                        ✅ Fully supports
                           Not accessible unless .retain_grad() used	
6)Gradients accumulation   ✅ Yes	                     ❌ No (returns fresh value)'''


#Example 3-> For nueral networks
# NOTE M.1 Manually

import torch
x=torch.tensor(6.7)
y=torch.tensor(0.)

w=torch.tensor(1.0)
b=torch.tensor(0.0)
#binary cross entropy losss
def binary_cross_entropy_loss(prediction,target):
    epsilon=1e-8#to prevent log (0)
    prediction=torch.clamp(prediction,epsilon,1-epsilon)
    return -(target*torch.log(prediction)+(1-target)*torch.log(1-prediction))

#Forward pass
z=w*x+b
y_pred=torch.sigmoid(z)

#compute loss
loss=binary_cross_entropy_loss(y_pred,y)

#Derivatives
#dl/dy_pred
dloss_dy_pred=(y_pred-y)/(y_pred*(1-y_pred))
#dy_pred/dz
dy_pred_dz=y_pred*(1-y_pred)
dz_dw=x
dz_db=1
dl_dw=dloss_dy_pred*dy_pred_dz*dz_dw
dl_db=dloss_dy_pred*dy_pred_dz*dz_db

print(f"Manual gradient of loss w.r.t weight is:{dl_dw}")
print(f"Manual gradient of loss w.r.t bias is:{dl_db}")

# NOTE M.2
x=torch.tensor(6.7)
y=torch.tensor(0.0)

w=torch.tensor(1.0,requires_grad=True)
b=torch.tensor(0.0,requires_grad=True)

z=w*x+b
y_pred=torch.sigmoid(z)
loss=binary_cross_entropy_loss(y_pred,y)
loss.backward()
print(w.grad)
print(b.grad)

#NOTE for vector imputs
x1=torch.tensor([1.,2.,3.],requires_grad=True)
y=(x1**2).mean()
y.backward()
print(x1.grad)


#NOTE For calculating higher order grads
import torch

# Step 1: Create input tensor with requires_grad=True
x = torch.tensor([1., 2., 3.], requires_grad=True)

# Step 2: Define a function of x
y = (x**2).mean()  # y = (x1^2 + x2^2 + x3^2) / 3

# Step 3: First derivative dy/dx
grad_y = torch.autograd.grad(y, x, create_graph=True)[0]

# Step 4: Second derivative d²y/dx²
second_derivatives = []

# Since grad_y is a vector, we take derivative of each component separately
for i in range(len(x)):
    second_deriv = torch.autograd.grad(grad_y[i], x, retain_graph=True)[0]
    second_derivatives.append(second_deriv)

# Stack results
hessian = torch.stack(second_derivatives)

print("First derivative dy/dx:", grad_y)
print("Second derivative d²y/dx² (Hessian):\n", hessian)

#NOTE Calling backward() many times leads to accumulation of gradients hence we prefer clearing the grads 
#so we must write before x.grad.zero_() at the end
x.grad.zero_()


#NOTE:
''' disable gradient tracking '''
'''  Options are:
                1)requires_grad=False//after the program ends 
                2)detach()
                3)torch.no_grad()
'''
#M.3->Best possible way
x2=torch.tensor(2.3,requires_grad=True)
with torch.no_grad():
    y=x**2

    

