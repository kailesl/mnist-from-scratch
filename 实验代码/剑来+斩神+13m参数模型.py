import sys
import os
import torch.nn.functional as F
import torch as torch

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.append("F:/ai/python_ai")
from torch_models.gpt_decoder import Transformer
tokenizer=AutoTokenizer.from_pretrained("bert-base-chinese")
def dataset_recieve():
    with open("F:/ai/python_ai/txt data/《剑+来+》（精校版全本）.txt","r",encoding="utf-8") as f:
        text=f.read()
        tokens=tokenizer.encode(text)
    with open("F:/ai/python_ai/txt data/校花贴身高手.txt","r",encoding="utf-8") as f:
        text1=f.read()
        #tokens1=tokenizer.encode(text1)
    with open("F:/ai/python_ai/txt data/斩神.txt","r",encoding="utf-8") as f:
        text2=f.read()
        tokens2=tokenizer.encode(text2)
    return tokens,tokens2
#token切分
tokens,tokens2=dataset_recieve()
max_len=256
data=[]
def tokenpro(tokens,tokens2):
    for i in range(0,len(tokens)-max_len,max_len):
        chunk=tokens[i:i+max_len]
        data.append(chunk)
    for i in range(0,len(tokens2)-max_len,max_len):
        chunk=tokens2[i:i+max_len]
        data.append(chunk)
    return data
data=tokenpro(tokens,tokens2)

vocab_size=len(tokenizer)
#数据处理
class GPTDataset(torch.utils.data.Dataset):
    def __init__(self,data):
        self.data=data
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        x=torch.tensor(self.data[idx])
        y=x.clone()
        y[:-1]=x[1:]
        y[-1]=-100
        return x,y
dataset=GPTDataset(data)
loader=DataLoader(dataset,batch_size=8,shuffle=True)

device="cuda" if torch.cuda.is_available() else "cpu"
model=Transformer(max_len,
                  vocab_size,
                  4,
                  4,
                  256).to(device)#(max_len,vocab_size,N,num_head,d_model)

#打印模型参数量
total_params = sum(
    p.numel()
    for p in model.parameters()
)
print(os.getcwd())

print(total_params)
optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)

def train(len):
    for epoch in range(len):
        for step,(x,y) in enumerate(loader):
            x=x.to(device)
            y=y.to(device)
            
            out=model(x)
            
            loss=F.cross_entropy(
                out.view(-1,vocab_size), #(batch,max_len,vocab_size)=(batch*max_len,vocab_size)
                y.view(-1), #(batch_size,max_len)=(batch_size*max_len) 对应每个词正确答案的下标
                ignore_index=-100
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step%100==0:
                print(f"epoch={epoch} | step={step} | loss={loss.item():.4f}")
            #按epoch来保存权重
            if step%500==0:
                torch.save(
                    {
                        "epoch":epoch,
                        "model":model.state_dict(),
                        "optimizer":optimizer.state_dict(),
                        "step":step
                    },
                    f"F:/ai/checkpoints/2026.5.31/ckpt_epoch{epoch}_step{step}.pth"
                )
train(8)
os.system("shutdown /s /t 0")