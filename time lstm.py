import numpy as np
class lstm:
    def __init__(self,hidden_state,word_length,lr):#hidden_state是隐藏状态向量的维度 word_length是词向量的维数
        self.lr=lr                       #列               行
        self.hidden_state=hidden_state
        self.weight_x=np.random.randn(word_length,4*hidden_state)*0.01
        self.weight_h=np.random.randn(hidden_state,4*hidden_state)*0.01#(-1,1)是转成列向量(1,-1)行向量
        self.bias=np.zeros((1,4*hidden_state)) 
        self.bias[:,: self.hidden_state]=1
    def sigmoid(self,value):
        return 1/(1+np.exp(-value))
    def tanh(self,value1):
        return (np.exp(value1)-np.exp(-value1))/(np.exp(value1)+np.exp(-value1))
    def forward(self,hidden_stat,c,word):
        self.state=hidden_stat
        self.c=c
        self.word=word
        #slice
        #self.forget=self.sigmoid(np.dot(self.word.reshape(1,-1),self.weight_x[:,:self.hidden_state])+np.dot(self.hidden_stat,self.weight_h[:,:self.hidden_state])+self.bias[:,:self.hidden_state])
        #()=(1,word)*(word,state)+(1,state)*(state,state)+(1,state)
        self.slice=np.dot(self.word.reshape(1,-1),self.weight_x)+np.dot(self.state,self.weight_h)+self.bias#(1,4state)=(1,word)*(word,4state)+(1,state)*(state,4state)+(1,4state)
        self.forget=self.sigmoid(self.slice[:,:self.hidden_state])
        self.memory=self.tanh(self.slice[:,self.hidden_state:2*self.hidden_state])
        self.input=self.sigmoid(self.slice[:,2*self.hidden_state:3*self.hidden_state])#这些维度都是(1,state)
        self.output=self.sigmoid(self.slice[:,3*self.hidden_state:4*self.hidden_state])
        self.c_new=self.c*self.forget+self.memory*self.input
        self.state_new=self.output*self.tanh(self.c_new)
        return self.state_new,self.c_new
    def backword(self,d_state1,d_state2,c_error):
        #梯度流动
        self.dc_new=c_error+self.output*(1-self.tanh(self.c_new)**2)*(d_state1+d_state2)
        self.d_output=self.tanh(self.c_new)*(d_state1+d_state2)
        self.d_input=self.memory*self.dc_new
        self.d_memory=self.input*self.dc_new
        self.d_forget=self.c*self.dc_new
        self.dc=self.forget*self.dc_new
        self.d_slice_forget=self.d_forget*self.forget*(1-self.forget)
        self.d_slice_memory=self.d_memory*(1-self.memory**2)
        self.d_slice_input=self.d_input*self.input*(1-self.input)
        self.d_slice_output=self.d_output*self.output*(1-self.output)#这上面都是(1,state)
        #self.d_slice=np.hstack(self.d_forget,self.d_memory,self.d_input,self.d_output)#hstack是吧向量从水平方向上拼起来，vstack是从竖直方向上拼起来
        self.d_slice=np.hstack((self.d_slice_forget,self.d_slice_memory,self.d_slice_input,self.d_slice_output))#(1,4state)
        #进行梯度流出
        self.dc_prev=self.dc
        self.d_state=np.dot(self.d_slice,self.weight_h.T)#(1,state)=(1,4state)*(4state,state)
        self.d_word=np.dot(self.d_slice,self.weight_x.T)
        #更新权重
        self.weight_x-=self.lr*np.dot(self.word.reshape(-1,1),self.d_slice)#(word,4state)=(word,1)*(1,4state)
        self.weight_h-=self.lr*np.dot(self.state.reshape(-1,1),self.d_slice)#(state,4state)=(state,1)*(1,4state)
        self.bias-=self.lr*self.d_slice
        return self.dc_prev,self.d_state,self.d_word
class timelstm:
    def __init__(self,hidden_state,wordlength,lr):
        self.hidden_state=hidden_state
        self.wordlength=wordlength
        self.lr=lr
        self.weight_x=np.random.randn(wordlength,4*hidden_state)*0.01
        self.weight_h=np.random.randn(hidden_state,4*hidden_state)*0.01
        self.bias=np.zeros((1,4*hidden_state))
        self.bias[:,:hidden_state]=1
        #self.lstm=lstm(self.hidden_state,self.wordlength,self.lr)
    def forward(self,sentence):
        self.sentence=sentence
        self.T=len(sentence)
        self.hidden_o=np.zeros((self.T,self.hidden_state))
        h=np.zeros((1,self.hidden_state))
        c=np.zeros((1,self.hidden_state))
        self.layer=[]
        for i in range(self.T):
            word=self.sentence[i]
            layer=lstm(self.hidden_state,self.wordlength,self.lr)
            #权重防止每个lstm都不一样
            layer.weight_x=self.weight_x
            layer.weight_h=self.weight_h
            layer.bias=self.bias
            h,c=layer.forward(h,c,word)
            self.hidden_o[i]=h
            self.layer.append(layer)#将每一个layer连接起来
        return self.hidden_o#lstm层的输出
    def backward(self,error_s):#error_s是上一层传过来的误差向量,维度是(T,state)
        self.error_hidden=np.zeros((self.T,self.hidden_state))
        dh=np.zeros((1,self.hidden_state))
        dc=np.zeros((1,self.hidden_state))
        for t in reversed(range(self.T)):#倒着循环  
            layer=self.layer[t]
            dc,dh,self.error_hidden[t]=layer.backward(error_s[t],dh,dc)
        return self.error_hidden