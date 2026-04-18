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
def to_one_hot(labels,num_classes):
    return np.eye(num_classes)[labels]
class word2vec:
    def __init__(self,input_output_node,hidden_node,lr):
        self.lr=lr
        self.input_o1=np.zeros(input_output_node)
        self.input_o2=np.zeros(input_output_node)
        self.hidden_oi=np.zeros(hidden_node)
        self.output_i=np.zeros(input_output_node)
        self.output_o=np.zeros(input_output_node)
        self.error_output=np.zeros(input_output_node)
        self.error_hidden=np.zeros(hidden_node)
        rng=np.random.default_rng(seed=702)#                  行数（列向量） 列数（行向量）
        self.weight_in1=rng.uniform(low=0.0,high=1.0,size=(input_output_node,hidden_node))
        self.weight_in2=rng.uniform(low=0.0,high=1.0,size=(input_output_node,hidden_node))#(-1,1)是转成列向量(1,-1)行向量
        self.weight_out=rng.uniform(low=0.0,high=1.0,size=(hidden_node,input_output_node))
    def softmax(self,x):
        max=np.max(x)
        self.x_exp=np.exp(x-max)
        self.sum=np.sum(self.x_exp)
        return (self.x_exp/self.sum).flatten() 
    def forward(self,word1,word2):
        self.input_o1=word1
        self.input_o2=word2
        self.hidden_oi=0.5*(np.dot(self.input_o1,self.weight_in1)+np.dot(self.input_o2,self.weight_in2))#这里有问题（ai改了）(hidden,1)=(hidden,input)*(input,1)  (hidden,1)hidden是列，（input，1）中input是列
        self.output_i=np.dot(self.weight_out.T,self.hidden_oi.reshape(-1,1))#(output,1)=(output,hidden)*(hidden,1)
        self.output_o=self.softmax(self.output_i)
    def backward(self,target):
        self.target=target
        loss=-np.sum(self.target*np.log(self.output_o+1e-8))
        self.error_output=self.output_o-self.target
        self.error_hidden=np.dot(self.weight_out,self.error_output.reshape(-1,1))#(hidden,1)=(hidden,output)*(output,1)
        self.weight_out-=self.lr*np.dot(self.hidden_oi.reshape(-1,1),self.error_output.reshape(1,-1))#(hidden,output) =(hidden,1)*(1,output)  为了使这里一一对齐我就没用转置等
        self.weight_in1-=self.lr*0.5*np.dot(self.input_o1.reshape(-1,1),self.error_hidden.reshape(1,-1),)#(input,hidden)=(input,1)*(1,hidden)
        self.weight_in2-=self.lr*0.5*np.dot(self.input_o2.reshape(-1,1),self.error_hidden.reshape(1,-1))
        return loss  
text_passage='Throughout our Junior year, my classmates and I have been worried about what colleges will see when they look at our whole life story reduced to a single 200-word essay. Will the golden word “success” form in their minds when they review our achievements? Or will they see the big word “fail” in red? The shadow of this mysterious institution steals away what success means to us.'
wt = wordstranslate()
wi,iw,co=wt.preprocess(text_passage)

context=[]
target_word=[]

for i in range(len(co)-2):
    context.append([
        to_one_hot(co[i],len(iw)),
        to_one_hot(co[i+2],len(iw))
    ])
    target_word.append(
        to_one_hot(co[i+1],len(iw))
    )

network=word2vec(len(iw),100,0.025)

for epoch in range(5):
    m=np.random.permutation(len(context))
    total_loss=0.0
    
    for i in m:
        m1=context[i][0]
        m2=context[i][1]
        t1=target_word[i]
        
        network.forward(m1,m2)
        total_loss+=network.backward(t1)
    
    print("epoch:",epoch,"loss:",total_loss/len(context))
#wi,iw,co=wordstranslate.preprocess(text_passage)
#context=[]
#target_word=[]
#network=word2vec(len(iw),100,0.025)
#for i in len(co)-2:
#    context[i]={to_one_hot(co[i],len(iw)),to_one_hot(co[i+2],len(iw))}
#    target_word[i]={to_one_hot(co[i+1],len(iw))}
#for epoch in range(5):
#    m=np.random.permutation(len(co)-2)
#    total_loss=0.0
#    for i in m:
#        m1=context[i][0]
#        m2=context[i][1]
#        t1=target_word[i]
#        network.forward(m1,m2)
#        total_loss+=network.backward(t1)