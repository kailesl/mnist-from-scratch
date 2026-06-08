import os
import sys
import torch.nn.functional as F
import torch as torch

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
sys.path.append("F:/ai/python_ai")
from torch_models.gpt_decoder import Transformer

tokenizer=AutoTokenizer.from_pretrained("bert-base-chinese")
#读取整个data文件夹
def get_txt_files(folder):
    txt_files=[]
    for file_name in os.listdir(folder):
        if file_name.endswith(".txt"):
            txt_files.append(
                os.path.join(
                    folder,
                    file_name
                )
            )
    return txt_files
#dataset处理+切块处理
class GPTDataset(torch.utils.data.Dataset):
    def __init__(self,tokens,max_len,stride=128):
        self.tokens=tokens
        self.max_len=max_len
        self.stride=stride
    def __len__(self):#这个是idx的长度
        return (len(self.tokens)-self.max_len)//self.stride
    def __getitem__(self,idx):
        start=idx*self.stride
        x=self.tokens[start:start+self.max_len]
        y=self.tokens[start+1:start+self.max_len+1]
        return torch.tensor(x,dtype=torch.long),torch.tensor(y,dtype=torch.long)

def load_checkpoint(path,model,optimizer,device):
    checkpoint=torch.load(path,map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch=checkpoint["epoch"]
    step=checkpoint["step"]
    print("成功加载checkpoint")
    return epoch,step

def train(epoch_num):
    #超参数设置
    device="cuda" if torch.cuda.is_available() else "cpu"
    max_len=256
    vocab_size=len(tokenizer)
    N=4
    num_head=4
    d_model=256
    model=Transformer(
        max_len,
        vocab_size,
        N,
        num_head,
        d_model
    ).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    #读取文件夹
    files=get_txt_files(
        "F:/ai/python_ai/txt data"
    )
    #读取checkpoints
    epoch,step=load_checkpoint(
        "F:/ai/checkpoints/2026.5.31/ckpt_epoch1_step1500.pth",
        model,
        optimizer,
        device
    )
    #读取文件
    tokens=[]
    for i,file in enumerate(files):
        with open(file,"r",encoding="utf-8") as f:
            text=f.read()
            text=tokenizer.encode(text)
            tokens.extend(text)
            print(f"text{i}")
    #dataset处理
    dataset=GPTDataset(tokens,max_len)
    loader=DataLoader(dataset,batch_size=8,shuffle=True)
    #打印模型参数量
    total_params=sum(p.numel()for p in model.parameters())
    print(total_params)
    
    for epoch in range(epoch_num):
        for step,(x,y) in enumerate(loader):
            x=x.to(device)
            y=y.to(device)
            
            out=model(x)
            
            loss=F.cross_entropy(
                out.view(-1,vocab_size),
                y.view(-1),
                ignore_index=-100
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step%100==0:
                print(f"epoch={epoch} | step={step} | loss={loss.item():.4f}")
            
            if step%1000==0:
                torch.save(
                    {
                        "epoch":epoch,
                        "model":model.state_dict(),
                        "optimizer":optimizer.state_dict(),
                        "step":step
                    },
                    f"G:/ai/checkpoints/gpt-13m-200mtoken-2026.6.6/ckpt_epoch{epoch}_step{step}.pth"
                )
train(8)
#注释代码区
#data切块
#def token_process(tokens,max_len):
#    data=[]
#    for i in range(0,len(tokens),max_len//2):
#        chunk=tokens[i:i+max_len]
#        data.append(chunk)
#    return data
 
#def file_read(folder,window_size):
#    with open(folder,"r",encoding="utf-8") as f:
#        text=f.read        