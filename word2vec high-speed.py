import numpy as np
class wordstranslate:
    def __init__(self):
        self.wordtoid={}
        self.idtoword={}
    def preprocess(self,text):
        self.text=text
        self.text=self.text.lower()#将词全部转化为小写
        self.text=self.text.replace('.',' .')#把所有的句号替换为空格加句号
        words=self.text.split(' ')
        for word in words:
            if word not in self.wordtoid:
                id=len(self.wordtoid)
                self.wordtoid[word]=id
                self.idtoword[id]=word
        corpus=[]
        for word in words:
            word_id=self.wordtoid[word]
            corpus.append(word_id) #把单词的id存进去
        return self.wordtoid,self.idtoword,corpus
class word2vec:
    def __init__(self,input_output_node,hidden_node,lr):
        self.lr=lr
        self.hidden=np.zeros(hidden_node)
        self.output_i=np.zeros(input_output_node)
        self.output_o=np.zeros(input_output_node)
        self.error_hidden=np.zeros(hidden_node)
        rng=np.random.default_rng(seed=702)#                    列                 行
        self.weight_in=rng.uniform(low=0.0,high=1.0,size=(input_output_node,hidden_node))
        self.weight_out=rng.uniform(low=0.0,high=1.0,size=(hidden_node,input_output_node))#(-1,1)是转成列向量(1,-1)行向量
    def sigmoid(self,value):
        return 1/(1+np.exp(-value))
    def NegativeSamplingforward(self,word1,word2,negsample,target):#word1与word2传送的是标签,都传的是标签 
        self.loss=0.0
        self.word1=word1
        self.word2=word2
        negsample=np.array(negsample)
        self.negsample=negsample
        self.target=target
        eps=1e-8
        context=word1+word2
        self.hidden=self.weight_in[context].mean(axis=0)#压缩向量，再乘以n分之一
        #正样本
        self.output_i=np.dot(self.hidden,self.weight_out[:,target])#提取target行 用正确样本去乘以错误样本的矩阵以达到负例学习的效果，我这行是不是错的
        self.output_o=self.sigmoid(self.output_i)
        self.output_target=self.output_o
        self.loss+=-np.log(self.output_o+eps)
        #负样本
        self.output_i=np.dot(self.hidden,self.weight_out[:,negsample])
        self.output_o=self.sigmoid(self.output_i)
        self.output_negsample=self.output_o
        self.loss+=-np.sum(np.log(1-self.output_o+eps))
    def backward(self):
        #正样本
        self.error_hidden=np.zeros_like(self.hidden)#清零error_hidden
        self.error_hidden+=(self.output_target-1)*self.weight_out[:,self.target] #上加号防止被覆盖掉
        self.error_hidden+=np.dot(self.weight_out[:,self.negsample],self.output_negsample.reshape(-1,1))#(hidden,1)=(hidden,negsample)*(negsample,1)
        self.weight_out[:,self.target]-=self.lr*(self.output_target-1)*self.hidden
        #负样本
        self.weight_out[:,self.negsample]-=self.lr*np.dot(self.hidden.reshape(-1,1),self.output_negsample.reshape(1,-1))#(hidden,negsample)=(hidden,1)*(1,negsmaple)
        self.weight_in[self.word1,:]-=1/(len(self.word1)+len(self.word2))*self.lr*self.error_hidden
        self.weight_in[self.word2,:]-=1/(len(self.word1)+len(self.word2))*self.lr*self.error_hidden
text=""
translate=wordstranslate()
word2vec1=word2vec()
iw,wi,co=translate.preprocess(text)
probability=np.zeros(len(iw))
word2vec1.__init__(len(iw),100,0.01)
for j in range(len(iw)):#因为iw是从0开始计数，那么它其实就等同于j
    js=0
    for i in range(len(co)):
        if j==co[i]:
            js+=1
    probability[j]=js/len(co)#把每个词的出现概率算出来 
probability_word=np.power(probability,0.75)
probability_word=probability_word/np.sum(probability_word)
window_size=5
for i in range(len(co)):
    if i<window_size or i>=len(co)-window_size:
        continue
    word1=co[i-window_size:i]
    word2=co[i+1:i+1+window_size]
    target=co[i]
    negsample=np.random.choice(len(iw),size=window_size,p=probability_word)
    word2vec1.NegativeSamplingforward(word1, word2, negsample, target) 
    word2vec1.backward()
print(word2vec1.loss)