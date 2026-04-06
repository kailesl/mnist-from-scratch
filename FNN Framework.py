import numpy as np
class Mnist:
    def __init__(self,input_node,hidden_node,output_node,lr):
        #初始化节点数目
        self.lr=lr
        self.input_node=input_node
        self.hidden_node=hidden_node
        self.output_node=output_node
        #创建并初始化节点
        self.hidden_i=np.zeros(hidden_node)
        self.hidden_o=np.zeros(hidden_node)
        self.output_i=np.zeros(output_node)
        self.output_o=np.zeros(output_node)
        #偏置初始化
        self.bias1=np.zeros(hidden_node)
        self.bias2=np.zeros(output_node)
        #创建一个随机数生成器
        rn=np.random.default_rng(seed=702)
        #根号n分之二的正态分布 relu专用
        w1=2/np.sqrt(input_node)
        w2=2/np.sqrt(hidden_node)
        self.weight1=rn.normal(loc=0,scale=w1,size=(input_node,hidden_node))
        self.weight2=rn.normal(loc=0,scale=w2,size=(hidden_node,output_node))#hidden_node是行数，
    def ReLU(self,value):
        return np.maximum(0,value)
    def ReLU_derivative(self,value1):
        #向量化写法
        return (value1>0).astype(float)#.astype(float)把前面的bool数组转成0和1的数组
    def softmax(self,value2):
        #防止出现nan
        ma=np.max(value2)
        exp_a=np.exp(value2-ma)
        exp_sum=np.sum(exp_a)
        return exp_a/exp_sum
    #前向传播
    def forward(self,inputs):
        self.input_i=inputs
        self.hidden_i=np.dot(inputs,self.weight1)+self.bias1
        self.hidden_o=self.ReLU(self.hidden_i)
        self.output_i=np.dot(self.hidden_o,self.weight2)+self.bias2
        self.output_o=self.softmax(self.output_i)
    def backward(self,target):
        self.error_output=target-self.output_o
        sum_loss=np.sum(self.error_output*self.error_output)
        self.error_hidden=np.dot(self.error_output,self.weight2.T)
        #reshape(1,-1)是转成行向量
        #reshape(-1,1)是转成列向量,一维向量没有行和列所以必须要转置
        #(a,1)*(1,b)=(a,b)
        #softmax的误差反向传播就是output-target，也就是没有求导的过程
        self.weight2-=-2*self.lr*np.dot(-self.error_output.reshape(-1,1),self.hidden_o.reshape(1,-1))
        self.bias2-=-2*self.lr*self.error_output#更新偏置
        #for i in range(0,self.output_node):
            #for j in range(0,self.hidden_node):
                #self.weight2[j][i]-=-2*self.lr*(-self.error_output[i])*self.hidden_o[j]
        #更新输入层到隐藏层的权重
        relu=self.ReLU_derivative(self.hidden_i)
        delta_hidden=-self.error_hidden*relu
        self.weight1-=-2*self.lr*np.dot(delta_hidden.reshape(-1,1),self.input_i.reshape(1,-1))
        self.bias1-=-2*self.lr*delta_hidden
        