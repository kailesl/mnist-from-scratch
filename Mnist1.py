import numpy as np
#
import os  
import urllib.request 

import struc
def load_images(filename):
    with open(os.path.join("data",filename),'rb')as f: 
        magic,num,rows,cols=struct.unpack(">IIII",f.read(16))
        images=np.frombuffer(f.read(),dtype=np.uint8).reshape(num,rows * cols)
        return images.astype(np.float32)/255.0
def load_labels(filename):
    with open(os.path.join("data", filename),'rb')as f:
        magic,num=struct.unpack(">II",f.read(8))
        labels=np.frombuffer(f.read(),dtype=np.uint8) 
        return labels
def to_one_hot(labels,num_classes=10):
    return np.eye(num_classes)[labels.flatten()] 
def load_mnist_data():
    train_images=load_images("train-images.idx3-ubyte")
    train_labels=load_labels("train-labels.idx1-ubyte")
    test_images=load_images("t10k-images.idx3-ubyte")
    test_labels=load_labels("t10k-labels.idx1-ubyte")
    return train_images,train_labels,test_images,test_labels
class Mnist:
    def __init__(self,input_node,hidden1_node,hidden2_node,output_node,lr):
        #初始化节点数目
        self.lr=lr
        self.input_node=input_node
        self.hidden1_node=hidden1_node
        self.hidden2_node=hidden2_node
        self.output_node=output_node
        #创建并初始化节点
        self.hidden1_i=np.zeros(hidden1_node)
        self.hidden1_o=np.zeros(hidden1_node)
        self.hidden2_i=np.zeros(hidden2_node)
        self.hidden2_o=np.zeros(hidden2_node)
        self.output_i=np.zeros(output_node)
        self.output_o=np.zeros(output_node)
        #偏置初始化
        self.bias1=np.zeros(hidden1_node)#第一层隐藏层偏置
        self.bias3=np.zeros(hidden2_node)#第二层隐藏层偏置
        self.bias2=np.zeros(output_node)#输出层偏置
        #创建一个随机数生成器
        rn=np.random.default_rng(seed=702)
        #根号n分之二的正态分布 relu专用
        w1=2/np.sqrt(input_node)
        w2=2/np.sqrt(hidden1_node)
        w3=2/np.sqrt(hidden2_node)
        self.weight1=rn.normal(loc=0,scale=w1,size=(input_node,hidden1_node))
        self.weight2=rn.normal(loc=0,scale=w2,size=(hidden1_node,hidden2_node))
        self.weight3=rn.normal(loc=0,scale=w3,size=(hidden2_node,output_node))
    def ReLU(self,value):
        return np.maximum(0,value)
    def ReLU_derivative(self,value1):
        return (value1>0).astype(float)#.astype(float)把前面的bool数组转成0和1的数组
    def softmax(self,value2):
        #防止出现nan
        ma=np.max(value2)
        exp_a=np.exp(value2-ma)
        exp_sum=np.sum(exp_a)
        return exp_a/exp_sum
    def forward(self,inputs):
        self.input_i=inputs.reshape(-1)#确保输入为一维向量784
        self.hidden1_i=np.dot(self.input_i,self.weight1)+self.bias1#第一隐藏层求和
        self.hidden1_o=self.ReLU(self.hidden1_i)#第一隐藏层输出
        self.hidden2_i=np.dot(self.hidden1_o,self.weight2)+self.bias3#第二隐藏层求和
        self.hidden2_o=self.ReLU(self.hidden2_i)#第二隐藏层输出
        self.output_i=np.dot(self.hidden2_o,self.weight3)+self.bias2#输出层求和
        self.output_o=self.softmax(self.output_i)#得到各类概率
    def backward(self,target):
        self.error_output=self.output_o-target
        sum_loss=-np.sum(target*np.log(self.output_o+1e-8))# 交叉熵误差，添加小的常数防止log(0)
        #输出层到hidden2
        self.error_hidden2=np.dot(self.error_output,self.weight3.T)
        #softmax的误差反向传播就是output-target，也就是没有求导的过程
        self.weight3-=self.lr*np.dot(self.hidden2_o.reshape(-1,1),self.error_output.reshape(1,-1))#weight是先行（隐藏层）后列（输出层）
        self.bias2-=self.lr*self.error_output
        #for i in range(0,self.output_node):
            #for j in range(0,self.hidden1_node):
                #self.weight2[j][i]-=-2*self.lr*(-self.error_output[i])*self.hidden_o[j]
        #hidden2到hidden1
        relu2=self.ReLU_derivative(self.hidden2_i)
        delta_hidden2=self.error_hidden2*relu2
        self.error_hidden1=np.dot(delta_hidden2,self.weight2.T)#误差继续传播到hidden1（必须用delta，不能用error）
        self.weight2-=self.lr*np.dot(self.hidden1_o.reshape(-1,1),delta_hidden2.reshape(1,-1))
        self.bias3-=self.lr*delta_hidden2
        #hidden到输入
        relu1=self.ReLU_derivative(self.hidden1_i)
        delta_hidden1=self.error_hidden1*relu1
        self.weight1-=self.lr*np.dot(self.input_i.reshape(-1,1),delta_hidden1.reshape(1,-1))
        self.bias1-=self.lr*delta_hidden1
        return sum_loss
trainx,trainy,testx,testy=load_mnist_data() #x是图片，y是数字标签
network=Mnist(784,512,256,10,0.01)
#开始训练
for epoch in range(5):
    #打乱数据顺序
    m=np.random.permutation(len(trainx))
    total_loss=0.0
    for i in m:
        img=trainx[i]
        number=trainy[i]
        target1=to_one_hot(np.array([number])).flatten()
        network.forward(img)
        total_loss+=network.backward(target1)
    correct=0
    for i in range(len(testx)):
        network.forward(testx[i])
        if np.argmax(network.output_o)==testy[i]:#argmax是求最大值的下标
            correct+=1
    print(f"epoch{epoch} loss:{total_loss/len(trainx):.4f} accuracy: {correct/len(testx)*100}%")
                 
