import numpy as np
class PositionEncoder:
    def __init__(self,seq_len,d_model):
        self.p=np.zeros((seq_len,d_model))
        self.d_model=d_model
        self.seq_len=seq_len
        
    def forward(self,x):#x=(seq_len,d_model)
        self.x=x
        seq_len=x.shape[0]
        self.p=np.zeros((seq_len,self.d_model))
        for i in range(seq_len):
            for j in range(self.d_model):
                if j%2==0 or j==0:
                    self.p[i][j]=np.sin(i/(10000**(j/self.d_model)))
                else:
                    self.p[i][j]=np.cos(i/(10000**(j/self.d_model)))
        self.an=self.x+self.p
        return self.an

class FeedForwardNetwork:
    def __init__(self,seq_len,d_model,lr):
        self.lr=lr
        self.seq_len=seq_len
        self.d_model=d_model
        self.d_ff=4*self.d_model
        self.w1=np.random.randn(d_model,self.d_ff)*np.sqrt(2/d_model)
        self.w2=np.random.randn(self.d_ff,d_model)*np.sqrt(2/d_model)
        self.b1=np.zeros((1,self.d_ff))
        self.b2=np.zeros((1,d_model))
    
    def relu(self,value):
        return np.maximum(0,value)
    
    def relu_f(self,value):
        return (value>0).astype(float)
        
    def forward(self,x):#x=(seq_len,d_model)
        self.x=x
        self.h=np.dot(x,self.w1)+self.b1#h=(seq_len,d_ff)
        self.ho=self.relu(self.h)
        self.output=np.dot(self.ho,self.w2)+self.b2#ho=(seq_len,d_model)
        return self.output
    
    def backward(self,do):#do=(seq_len,d_model)
        self.dho=np.dot(do,self.w2.T)#dho=(seq_len,d_ff)
        self.b2-=self.lr*do
        self.w2-=self.lr*np.dot(self.ho.T,do)
        self.dh=self.relu_f(self.h)*self.dho#dh=(seq_len,d_ff)
        self.dx=np.dot(self.dh,self.w1.T)
        self.b1-=self.lr*self.dh
        self.w1-=self.lr*np.dot(self.x.T,self.dh)
        return self.dx
        
class MultiHeadAttention:
    def __init__(self,d_model,num_head,seq_len1,seq_len2,lr):#seq_len1为本地句子,seq_len2为外部输入,如果是在编码其中直接让seq1=seq2相等即可
        assert d_model%num_head==0
        self.d_model=d_model
        self.num_head=num_head
        self.seq_len1=seq_len1
        self.seq_len2=seq_len2
        self.lr=lr
        self.w_q=np.random.randn(d_model,d_model)
        self.w_k=np.random.randn(d_model,d_model)#d_k=d_model/num_head
        self.w_v=np.random.randn(d_model,d_model)#w_q=(d_model,d_model/num_head=d_k)
        self.w_out=np.random.randn(d_model,d_model)
        self.d_k=d_model//num_head
        
    def softmax(self,value):#所有都会触发广播
        ma=np.max(value,axis=-1,keepdims=True)#沿着前一个seq压缩 max把每行中的最大值提取出来让后直接压缩
        value1=np.exp(value-ma)
        sum=np.sum(value1,axis=-1,keepdims=True)#keepdims保持维度
        return value1/sum
    
    def softmax_f(self,value):
        soft=self.softmax(value)
        return soft*(1-soft)
    
    def forward(self,x,y,mask):#x=(seq_len1,d_model) x为本地输入，y为外地输入
        self.x=x
        self.y=y#y=(seq_len2,d_model)
        self.mask=mask#mask=(seq_len1,seq_len2)
        
        self.wq=self.w_q.reshape(self.d_model,self.num_head,self.d_k)#切开成为(d_model,num_head,d_k)
        self.wq=self.wq.transpose(1,0,2)#转置成为(num_head,d_model,d_k)
        
        self.wk=self.w_k.reshape(self.d_model,self.num_head,self.d_k)
        self.wk=self.wk.transpose(1,0,2)
        
        self.wv=self.w_v.reshape(self.d_model,self.num_head,self.d_k)
        self.wv=self.wv.transpose(1,0,2)#wv=(num,d_model,d_k)
        
        self.q=np.matmul(self.x[None,:,:],self.wq)#q=(num_head,seq_len1,d_k),x=(1,seq_len1,d_model)
        if y is not None:
            self.k=np.matmul(self.y[None,:,:],self.wk)#k=(num_head,seq_len2,d_k)
            self.v=np.matmul(self.y[None,:,:],self.wv)#v=(num_head,seq_len2,d_k)
        else:
            self.k=np.matmul(self.x[None,:,:],self.wk)
            self.v=np.matmul(self.x[None,:,:],self.wv)
        
        self.at=np.matmul(self.q,self.k.transpose(0,2,1))/np.sqrt(self.d_k)
        if mask is not None:
            seq1l=self.x.shape[0]
            seq2l=self.y.shape[0] if self.y is not None else seq1l
            self.mask_act=mask[:seq1l,:seq2l]  
            self.at=self.at+self.mask_act
        self.score=self.softmax(self.at)#score=(num_head,seq1,seq2)
        self.attention=np.matmul(self.score,self.v)#attention=(num_head,seq_len1,d_k)
        
        self.attention=self.attention.transpose(1,0,2)
        self.seql=self.x.shape[0]
        self.attention=self.attention.reshape(self.seql,self.d_model)
        self.output=np.dot(self.attention,self.w_out)#output=(seq_len1,d_model)
        return self.output
    
    def backward(self,da):#da=(seq_len1,d_model)
        mh=da
        da=np.dot(da,self.w_out.T)
        self.w_out-=self.lr*np.dot(self.attention.T,mh)
        
        self.seql=da.shape[0]
        self.da=da.reshape(self.seql,self.d_k,self.num_head)
        self.da=self.da.transpose(2,0,1)#da=(num,seq1,d_k)
        
        self.ds=np.matmul(self.da,self.v.transpose(0,2,1))
        self.dv=np.matmul(self.score.transpose(0,2,1),self.da)#dv=(num_head,seq2,d_k)
        self.m=self.softmax_f(self.at)*self.ds#at=(num_head,seq_len1,seq_len2) ds=(num,seq_len1,seq_len2)
        
        self.dq=np.matmul(self.m,self.k/np.sqrt(self.d_k))# k=(num,seq2,d_k)
        self.dk=np.matmul(self.m.transpose(0,2,1),self.q/np.sqrt(self.d_k))
        
        self.dwq=np.matmul(self.x[None,:,:].transpose(0,2,1),self.dq)#dw=(num,d_model,d_k)
        self.dwq=self.dwq.transpose(1,0,2)#dw=(d_model,num,d_k)
        self.dx=np.zeros_like(self.x)    
        if self.y is not None:
            self.dy=np.zeros_like(self.y) 
        else:
            self.dy=None
        self.dx+=np.sum(np.matmul(self.dq,self.wq.transpose(0,2,1)),axis=0)
        
        if self.y is not None:
            self.dwv=np.matmul(self.y[None,:,:].transpose(0,2,1),self.dv)
            self.dwk=np.matmul(self.y[None,:,:].transpose(0,2,1),self.dk)
            self.dwv=self.dwv.transpose(1,0,2)
            self.dwk=self.dwk.transpose(1,0,2)
            
            self.dy+=np.sum(np.matmul(self.dk,self.wk.transpose(0,2,1)),axis=0)
            self.dy+=np.sum(np.matmul(self.dv,self.wv.transpose(0,2,1)),axis=0)#dy=(num,seq2,d_k)*(num,d_k,d_model)=(num,seq2,d_model)
        else:
            self.dwv=np.matmul(self.x[None,:,:].transpose(0,2,1),self.dv)
            self.dwk=np.matmul(self.x[None,:,:].transpose(0,2,1),self.dk)
            self.dwv=self.dwv.transpose(1,0,2)
            self.dwk=self.dwk.transpose(1,0,2)
        
            self.dx+=np.sum(np.matmul(self.dk,self.wk.transpose(0,2,1)),axis=0)
            self.dx+=np.sum(np.matmul(self.dv,self.wv.transpose(0,2,1)),axis=0)
        
        self.w_q-=self.lr*self.dwq.reshape(self.d_model,self.d_model)
        self.w_k-=self.lr*self.dwk.reshape(self.d_model,self.d_model)
        self.w_v-=self.lr*self.dwv.reshape(self.d_model,self.d_model)
        return self.dx,self.dy

class Norm:#使每层的分布更加平均
    def __init__(self,d_model,lr,eps=1e-6):
        self.w=np.ones((d_model))#设为一的原因是为了让第一层的层规范不消失
        self.b=np.zeros((d_model))
        self.eps=eps
        self.lr=lr
        self.d_model=d_model
    
    def forward(self,x):#x=(seq1,d_model)
        self.x=x
        self.aver=np.mean(x,axis=-1,keepdims=True)#取每层的平均值a=(d_model)
        self.var=np.var(x,axis=-1,keepdims=True)#取每层的方差v=(d_model)
        self.m=(x-self.aver)/np.sqrt(self.var+self.eps)#层规范化+防爆m=(seq1,d_model)
        self.out=self.w*self.m+self.b#让神经网络自己学
        return self.out
        
    def backward(self,do):#do=(seq1,d_model)
        self.b-=self.lr*np.sum(do,axis=0)
        self.dm=do*self.w
        self.w-=self.lr*np.sum(do*self.m,axis=0) 
        
        self.dx=self.dm/np.sqrt(self.var+self.eps)
        self.d_aver=-self.dm/np.sqrt(self.var+self.eps)#d_aver=(seq_len,d_model)
        self.dx1=np.mean(self.d_aver,axis=-1,keepdims=True)
        
        self.d_var=-0.5*np.sum(self.dm*(self.x-self.aver),axis=-1,keepdims=True)/(self.var**1.5)#d_var=(seq_len,d_model)
        self.dx2=(2.0/self.d_model)*(self.x-self.aver)*self.d_var
        
        self.dx=self.dx+self.dx2+self.dx1
        return self.dx

class encoder:
    def __init__(self,seq_len,d_model,num_head,lr):
        self.mha=MultiHeadAttention(d_model,num_head,seq_len,seq_len,lr)
        self.ffn=FeedForwardNetwork(seq_len,d_model,lr)
        self.normlayer1=Norm(d_model,lr)
        self.normlayer2=Norm(d_model,lr)
    
    def forward(self,input):
        #过多头注意力层
        self.o1=self.mha.forward(input,None,None)
        self.o2=self.normlayer1.forward(self.o1+input)
        #过前馈
        self.o3=self.ffn.forward(self.o2)
        self.o4=self.normlayer2.forward(self.o3+self.o2)
        return self.o4
    
    def backward(self,do):#变量名标识为梯度的去向
        #过前馈
        self.do3=self.normlayer2.backward(do)
        self.do2=self.ffn.backward(self.do3)
        #过多头注意力层
        self.do1=self.normlayer1.backward(self.do2+self.do3)
        self.di,_=self.mha.backward(self.do1)
        
        self.di=self.di+self.do1
        return self.di

class decoder:
    def __init__(self,seq_len1,seq_len2,d_model,num_head,lr):#seq_len1为本地句子,seq_len2为外部输入
        self.mask=np.triu(np.ones((seq_len1,seq_len2))*-1e9,k=1)#k=1表示保留对角线上方的元素，将对角线及其以下的部分变为零
        self.mmha=MultiHeadAttention(d_model,num_head,seq_len1,seq_len1,lr)#掩码多头
        self.normlayer1=Norm(d_model,lr)
        self.cmha=MultiHeadAttention(d_model,num_head,seq_len1,seq_len2,lr)#交互多头
        self.normlayer2=Norm(d_model,lr)
        self.ffn=FeedForwardNetwork(seq_len1,d_model,lr)
        self.normlayer3=Norm(d_model,lr)
    
    def forward(self,input,encoder_in):
        #过掩码多头
        self.mmha_o=self.mmha.forward(input,None,self.mask)
        self.nl1_o=self.normlayer1.forward(input+self.mmha_o)
        #过交互多头
        self.cmha_o=self.cmha.forward(self.nl1_o,encoder_in,None)
        self.nl2_o=self.normlayer2.forward(self.nl1_o+self.cmha_o)
        #过前馈
        self.ffn_o=self.ffn.forward(self.nl2_o)
        self.nl3_o=self.normlayer3.forward(self.nl2_o+self.ffn_o)
        return self.nl3_o
    
    def backward(self,do):
        #过前馈
        self.d1=self.normlayer3.backward(do)
        self.dffn=self.ffn.backward(self.d1)
        #过交互多头
        self.d2=self.normlayer2.backward(self.d1+self.dffn)
        self.dcmha,self.dencoder_in=self.cmha.backward(self.d2)
        #过掩码多头
        self.d3=self.normlayer1.backward(self.d2+self.dcmha)
        self.dmmha,_=self.mmha.backward(self.d3)
        
        self.di=self.d3+self.dmmha
        return self.di,self.dencoder_in

class Transformer:
    def __init__(self,tgt_vocab,d_model,num_head,num_layers,max_seq,lr):#max_seq是模型最多能处理的序列长度,src与tgt都是词表大小 需要建两个word2vec的词表
        self.lr=lr
        self.num_layers=num_layers
        self.encoder=[encoder(max_seq,d_model,num_head,lr)for _ in range(num_layers)]
        self.decoder=[decoder(max_seq,max_seq,d_model,num_head,lr)for _ in range(num_layers)]
        self.encoder_pe=PositionEncoder(max_seq,d_model)
        self.decoder_pe=PositionEncoder(max_seq,d_model)
        self.w_out=np.random.randn(d_model,tgt_vocab)
    
    def forward(self,x,y):#x是编码器的输入(用word2vec处理好的)，y是解码器的输入，也就是目标序列的前一位
        self.x=self.encoder_pe.forward(x)
        self.y=self.decoder_pe.forward(y)
        
        self.output_encoder=self.x
        for layer in self.encoder:
            self.output_encoder=layer.forward(self.output_encoder)
        
        self.output_decoder=self.y    
        for layer in self.decoder:
            self.output_decoder=layer.forward(self.output_decoder,self.output_encoder)
        
        self.output=np.dot(self.output_decoder,self.w_out)#output=(seq,tgt_vocab)  其实按照常理来说seq也是target也就是翻译后的语言词表
        return self.output
    
    def backward(self,dl):
        self.d_dec=np.dot(dl,self.w_out.T)#(seq,d_model)
        self.w_out-=self.lr*np.dot(self.output_decoder.T,dl)#w=(d_model,tgt_vocab)
        
        self.d_enc_total=np.zeros_like(self.d_dec)
        for layer in reversed(self.decoder):
            self.d_dec,self.d=layer.backward(self.d_dec)
            self.d_enc_total+=self.d
           
        self.d_enc=self.d_enc_total 
        for layer in reversed(self.encoder):
            self.d_enc=layer.backward(self.d_enc)
        return self.d_enc

def softmax_cross_entropy(model_output,target):#模型输出和target,  target为目标词的位置序列
    ma=np.max(model_output,axis=-1,keepdims=True)
    ex=np.exp(model_output-ma)
    output=ex/np.sum(ex,axis=-1,keepdims=True)
    
    seq_len=model_output.shape[0]
    p=output[np.arange(seq_len),target] #词表中寻找正确词的位置
    loss=-np.mean(np.log(p+1e-9))#求平均误差
    
    d=output.copy()
    d[np.arange(seq_len),target]-=1
    return d,loss#d=(seq,target)
