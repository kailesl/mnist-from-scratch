import numpy as np
class node:
    def __init__(self,lr,hidden_number,hidden_node):
        self.lr=lr
        self.h_i=np.zeros((hidden_node,hidden_number))
        self.h_o=np.zeros((hidden_node,hidden_number))
        self.w=np.random.randn(hidden_node,hidden_node,hidden_number)*0.01#(前一层节点数，后一层节点数，隐藏层数)
    def relu(self,value):
        return np.maximum(0,value)
    def forward(self,h_o,i):
        h_o=h_o.reshape(-1)
        self.h_i[:,i]=np.dot(self.w[:,:,i],h_o)
        self.h_o[:,i]=self.relu(self.h_i[:,i])
        return self.h_o[:,i]
    def relu_f(self,value):
        return (value>0).astype(float)
    def backward(self,dh,j,input):#当j=0时这里可能有问题 
        self.dh_i=dh*self.relu_f(self.h_i[:,j])#(hidden,1)=(hidden,1)*(hidden,1)
        self.dh_s=np.dot(self.w[:,:,j].T,self.dh_i)#(hidden_后,1)=(hidden_后,hidden_前)*(hidden_前,1)
        if j>=1:
            self.w[:,:,j]-=self.lr*np.dot(self.dh_i.reshape(-1,1),self.h_o[:,j-1].reshape(1,-1))#(hidden_前,hidden_后)=(hidden_前,1)*(1,hidden_后)
        else:
            self.w[:,:,j]-=self.lr*np.dot(self.dh_i.reshape(-1,1),input.reshape(1,-1))
        return self.dh_s
class mlp:
    def __init__(self,lr,start_node,last_node,hidden_number,hidden_node):
        self.lr=lr
        self.hidden_number=hidden_number
        self.input=np.zeros((start_node))
        self.output_i=np.zeros((last_node))
        self.output_o=np.zeros((last_node))
        self.w_output=np.random.randn(hidden_node,last_node)
        self.hidden=node(self.lr,hidden_number,hidden_node)
    def softmax(self,value1):
        mx=np.max(value1)
        value2=np.exp(value1-mx)
        sum=np.sum(value2)
        return value2/sum
    def forward(self,input):
        self.input=input
        h=[]
        for i in range(self.hidden_number):
            if i==0:
                h=self.hidden.forward(self.input,i)
            else:
                h=self.hidden.forward(h,i)#到时候从这里改
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
                dh=self.hidden.backward(d1,j,self.input)
            else:
                dh=self.hidden.backward(dh,j,self.input)
        return self.loss