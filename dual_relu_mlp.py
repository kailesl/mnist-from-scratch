import numpy as np
class node:
    def __init__(self,lr,hidden_number,hidden_node):
        self.lr=lr
        self.hidden_node=hidden_node
        self.hidden_number=hidden_number
        
        self.input=np.zeros((hidden_node,hidden_number))
        
        self.h_i1=np.zeros((hidden_node,hidden_number))
        self.h_i2=np.zeros((hidden_node,hidden_number))
        self.h_i3=np.zeros((hidden_node,hidden_number))
        self.h_i4=np.zeros((hidden_node,hidden_number))
        
        self.h_o1=np.zeros((hidden_node,hidden_number))
        self.h_o2=np.zeros((hidden_node,hidden_number))
        self.h_o3=np.zeros((hidden_node,hidden_number))
        self.h_o4=np.zeros((hidden_node,hidden_number))
        
        self.wq_1=np.random.randn(hidden_node,hidden_node,hidden_number)*np.sqrt(2/hidden_node)
        self.w1_2=np.random.randn(hidden_node,hidden_node,hidden_number)*np.sqrt(2/hidden_node)
        self.w2_3=np.random.randn(hidden_node,hidden_node,hidden_number)*np.sqrt(2/hidden_node)#(hidden后,hidden前)
        self.wq_4=np.random.randn(hidden_node,hidden_node,hidden_number)*np.sqrt(2/hidden_node)
        self.w4_3=np.random.randn(hidden_node,hidden_node,hidden_number)*np.sqrt(2/hidden_node)
    def relu(self,value):
        return np.maximum(0,value)
    def relu_f(self,value):
        return (value>0).astype(float)
    def forward(self,h_o,i):
        h_o=h_o.reshape(self.hidden_node,)
        self.input[:,i]=h_o
        #分支一
        self.h_i1[:,i]=np.dot(self.wq_1[:,:,i],h_o)#(hidden后,1)=(hidden后,hidden前)*(hidden前,1)
        self.h_o1[:,i]=self.relu(self.h_i1[:,i])
        
        self.h_i2[:,i]=np.dot(self.w1_2[:,:,i],self.h_o1[:,i])#
        self.h_o2[:,i]=self.relu(self.h_i2[:,i])#
        #分支二
        self.h_i4[:,i]=np.dot(self.wq_4[:,:,i],h_o)#
        self.h_o4[:,i]=self.relu(self.h_i4[:,i])#
        #汇入
        self.h_i3[:,i]=0
        self.h_i3[:,i]+=np.dot(self.w2_3[:,:,i],self.h_o2[:,i])#
        self.h_i3[:,i]+=np.dot(self.w4_3[:,:,i],self.h_o4[:,i])
        #残差汇入
        self.h_i3[:,i]+=h_o
        self.h_o3[:,i]=self.relu(self.h_i3[:,i])
        return self.h_o3[:,i]
    def backward(self,dh,j):
        self.ds=np.zeros((self.hidden_node))
        #梯度流入
        self.dh_i3=dh*self.relu_f(self.h_i3[:,j])
        #残差的这条路
        self.ds+=self.dh_i3
        #分支二
        self.dh_o4=np.dot(self.w4_3[:,:,j].T,self.dh_i3)#(hidden前,1)=(hidden前,hidden后)*(hidden后,1)
        self.w4_3[:,:,j]-=self.lr*np.dot(self.dh_i3.reshape(-1,1),self.h_o4[:,j].reshape(1,-1))#(hidden后,hidden前)=(hidden后,1)*(1,hidden前)
        self.dh_i4=self.dh_o4*self.relu_f(self.h_i4[:,j])
        self.ds+=np.dot(self.wq_4[:,:,j].T,self.dh_i4)
        #分支一
        self.wq_4[:,:,j]-=self.lr*np.dot(self.dh_i4.reshape(-1,1),self.input[:,j].reshape(1,-1))
        self.dh_o2=np.dot(self.w2_3[:,:,j].T,self.dh_i3)
        self.w2_3[:,:,j]-=self.lr*np.dot(self.dh_i3.reshape(-1,1),self.h_o2[:,j].reshape(1,-1))
        self.dh_i2=self.dh_o2*self.relu_f(self.h_i2[:,j])
        self.dh_o1=np.dot(self.w1_2[:,:,j].T,self.dh_i2)
        self.w1_2[:,:,j]-=self.lr*np.dot(self.dh_i2.reshape(-1,1),self.h_o1[:,j].reshape(1,-1))
        self.dh_i1=self.dh_o1*self.relu_f(self.h_i1[:,j])
        self.ds+=np.dot(self.wq_1[:,:,j].T,self.dh_i1)
        self.wq_1[:,:,j]-=self.lr*np.dot(self.dh_i1.reshape(-1,1),self.input[:,j].reshape(1,-1))
        return self.ds
class mlp:
    def __init__(self,lr,start_node,last_node,hidden_number,hidden_node):
        self.lr=lr
        self.hidden_number=hidden_number
        self.input=np.zeros((start_node))
        self.output_i=np.zeros((last_node))
        self.output_o=np.zeros((last_node))
        
        self.w_output=np.random.randn(hidden_node,last_node)*np.sqrt(2/hidden_node)
        self.w_input=np.random.randn(start_node,hidden_node)*np.sqrt(2/start_node)
        
        self.hidden=node(self.lr,hidden_number,hidden_node)
        
    def softmax(self,value1):
        mx=np.max(value1)
        value2=np.exp(value1-mx)
        sum=np.sum(value2)
        return value2/sum
    
    def relu(self,value):
        return np.maximum(0,value)
    
    def relu_f(self,value):
        return (value>0).astype(float)
    
    def forward(self,input):
        self.raw_input=input
        self.m=np.dot(input,self.w_input)
        self.input=self.relu(self.m)
        h=self.input
        
        for i in range(self.hidden_number):
            h=self.hidden.forward(h,i)
        self.h_last=h
        self.output_i=np.dot(self.w_output.T,h)#(last_node,1)=(last,hidden)*(hidden,1)
        self.output_o=self.softmax(self.output_i)
        
    def backward(self,target):
        self.loss=-np.sum(target*np.log(self.output_o+1e-8))
        
        d=self.output_o-target
        
        d1=np.dot(self.w_output,d.reshape(-1,1)).reshape(-1)#(hidden,1)=(hidden,last)*(last,1)
        
        self.w_output-=self.lr*np.dot(self.h_last.reshape(-1,1),d.reshape(1,-1))
        
        dh=[]
        
        for j in reversed(range(self.hidden_number)):
            if j==self.hidden_number-1:
                dh=self.hidden.backward(d1,j)
            else:
                dh=self.hidden.backward(dh,j)
        dm=dh*self.relu_f(self.m)
        self.w_input-=self.lr*np.dot(self.raw_input.reshape(-1,1),dm.reshape(1,-1))        
        return self.loss