import os
import sys
import torch.nn.functional as F
import torch as torch
import random
import time

from transformers import AutoTokenizer
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
sys.path.append("F:/ai/python_ai/torch_models")
from torch_models.gpt_decoder import Transformer

tokenizer=AutoTokenizer.from_pretrained("Qwen2.5-0.5B")
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
#dataset处理+流式切块
class tokenstreamdataset(IterableDataset):
    def __init__(self,tokenizer,files,max_len,stride=128,chunk_size=200000):
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.files=files
        self.stride=stride
        self.chunk_size=chunk_size
        self.buffer_size=10000
    def __iter__(self):
        shuffle_buffer=[]
        token_buffer=[]
        for file in self.files:
            with open(file,"r",encoding="utf-8") as f:
                while True:
                    text=f.read(self.chunk_size)
                    if not text:
                        break
                    tokens=self.tokenizer.encode(text)
                    token_buffer.extend(tokens)
                    while len(token_buffer)>=self.max_len+1:
                        x=token_buffer[:self.max_len]
                        y=token_buffer[1:self.max_len+1]
                        shuffle_buffer.append((x,y))
                        if len(shuffle_buffer)>=self.buffer_size:
                            idx=random.randrange(len(shuffle_buffer))
                            fo,g=shuffle_buffer.pop(idx)
                            yield torch.tensor(fo,dtype=torch.long),torch.tensor(g,dtype=torch.long)
                        token_buffer=token_buffer[self.stride:]
        random.shuffle(shuffle_buffer)
        while shuffle_buffer:
            fo,g=shuffle_buffer.pop()
            yield torch.tensor(fo,dtype=torch.long),torch.tensor(g,dtype=torch.long)
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
    scaler=torch.amp.GradScaler("cuda")
    #读取文件夹
    files=get_txt_files(
        "F:/ai/python_ai/txt data"
    )
    #读取checkpoints
    epoch,step=load_checkpoint(
        "G:/ai/checkpoints/gpt-13m-200mtoken-2026.6.6/ckpt_epoch0_step8000.pth",
        model,
        optimizer,
        device
    )
    #dataset处理
    dataset=tokenstreamdataset(tokenizer,files,max_len)
    loader=DataLoader(dataset,batch_size=16,num_workers=2,pin_memory=True)
    #打印模型参数量
    total_params=sum(p.numel()for p in model.parameters())
    print(total_params)
    
    for epoch in range(epoch_num):
        for step,(x,y) in enumerate(loader):
            x=x.to(device)
            y=y.to(device)
            with torch.amp.autocast("cuda"):
                out=model(x)
                
                loss=F.cross_entropy(
                    out.view(-1,vocab_size),
                    y.view(-1),
                    ignore_index=-100
                )
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
                    f"G:/ai/checkpoints/流式预训练2026.6.6/ckpt_epoch{epoch}_step{step}.pth"
                )

def generate(temperature,top_k):
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
    #读取checkpoints
    epoch,step=load_checkpoint(
        "C:/Users/Administrator/Desktop/云端bag/checkpoints/ckpt_epoch6_step19135.pth",
        model,
        optimizer,
        device
    )
    model.eval()
    
    prompt=input("请输入请输入提示词：")
    prompt=tokenizer.encode(prompt)
    for i in range(max_len-len(prompt)):
        b=torch.tensor(prompt,dtype=torch.long,device=device).unsqueeze(0)
        with torch.no_grad():
            out=model(b)#out=(batch,seq,vocab_size)
            out=out[:,-1,:]
            out=out/temperature#out=(batch,vocab_size)
            score=F.softmax(out,dim=-1)
            topk_token,topk_id=torch.topk(score,top_k,dim=-1)#toke_id是词的下标
            idx=torch.multinomial(topk_token,1)
            next_token=topk_id.gather(-1,idx)
            next_token=next_token.item()
            print(tokenizer.decode(next_token,add_special_tokens=False),end="")
            time.sleep(0.05)
            prompt.append(next_token)
generate(1.1,30)
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
        
    # for epoch in range(epoch_num):
    #     for step,(x,y) in enumerate(loader):
    #         x=x.to(device)
    #         y=y.to(device)
            
    #         out=model(x)
            
    #         loss=F.cross_entropy(
    #             out.view(-1,vocab_size),
    #             y.view(-1),
    #             ignore_index=-100
    #         )
            
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         if step%100==0:
    #             print(f"epoch={epoch} | step={step} | loss={loss.item():.4f}")
            
    #         if step%1000==0:
    #             torch.save(
    #                 {
    #                     "epoch":epoch,
    #                     "model":model.state_dict(),
    #                     "optimizer":optimizer.state_dict(),
    #                     "step":step
    #                 },
    #                 f"G:/ai/checkpoints/gpt-13m-200mtoken-2026.6.6/ckpt_epoch{epoch}_step{step}.pth"
    #             )
    
    # #读取文件
    # tokens=[]
    # for i,file in enumerate(files):
    #     with open(file,"r",encoding="utf-8") as f:
    #         text=f.read()
    #         text=tokenizer.encode(text)
    #         tokens.extend(text)
    #         print(f"text{i}")
    # #dataset处理
    # dataset=GPTDataset(tokens,max_len)
    # loader=DataLoader(dataset,batch_size=8,shuffle=True)这样子没问题吧