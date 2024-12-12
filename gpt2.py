# torchrun --standalone --nproc_per_node=8 
from dataclasses import dataclass
import os
import time 
import torch
import torch.nn as nn
from torch.nn import functional as F 
from torch.distributed import init_process_group,destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import GPT2LMHeadModel
import math
import tiktoken
import sys
import numpy as np 

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt,dtype=torch.long)
    return ptt

class DataLoaderlite:
    def __init__(self,B,T,process_rank,num_processes,split):
        self.B = B
        self.T = T 
        self.process_rank  = process_rank
        self.num_processes = num_processes
        assert split in {'train','val'}

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s ]
        self.shards = shards
        assert len(shards) > 0
        if master_process:
            print(f"found:{len(shards)} shards for split:{split}")
        self.reset()


    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank


    def next_batch(self):
        B,T = self.B,self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        self.current_position += B*T * self.num_processes
        if self.current_position + (B*T * self.num_processes + 1)>len(self.tokens):
            self.current_shard = (self.curent_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x,y 





class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd,3*config.n_embd)
        #
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).
                             view(1,1,config.block_size,config.block_size))
        
    def forward(self,x):
        B,T,C = x.size()
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd,dim=2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)

       # att = (q@k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
       
       # att = att.masked_fill(self.bias[:,:,:T,:T] == 0,float('-inf'))
       # att = F.softmax(att,dim=-1)
       # y = att @ v
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass 
class GPTConfig:
    block_size:int = 1024
    vocab_size:int = 50257
    n_layer:int = 12
    n_head:int = 12
    n_embd:int = 768


class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
            
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.transformer.wte.weight = self.lm_head.weight 
        self.apply(self.__init__weights)

    def __init__weights(self,module):
        if isinstance(module,nn.Linear):
            std = 0.02
            if hasattr(module,'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
 

    def forward(self,idx,targets=None):
        B,T = idx.size()
        assert T<= self.config.block_size, f"canot forward sequence of length{T}"
        pos = torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb 

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits,loss



    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
           
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
  
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self,weight_decay,learning_rate,device):
        param_dict = {pn: p for pn,p in self.named_parameters()}
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}
        decay_params = [p for n,p in param_dict.items()if p.dim() >= 2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params':decay_params,'weight_decay':weight_decay},
            {'params':nodecay_params,'weight_decay':0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        optimizer = torch.optmi.Adamw(optim_groups,lr=learning_rate,betas = (0.9,0.95),eps=1e-8,fused=use_fused)
        return optimizer




ddp = int(os.environ.get("RANK",-1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank ==0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device:{device}")



   

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
    
total_batch_size = 524288
B = 64
T = 1024
assert total_batch_size % (B*T*ddp_world_size) == 0,"make sure total batch size is divisible by B*T"
grad_accum_steps  = total_batch_size // (B*T*ddp_world_size)


print(f"i am gpu:{ddp_rank}")


train_loader = DataLoaderlite(B=B,T=T,process_rank=ddp_rank,num_processes=ddp_world_size,split="train")
val_loader = DataLoaderlite(B=B,T=T,process_rank=ddp_rank,num_processes=ddp_world_size,split="val")

torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)

if ddp:
    model = DDP(model,device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1)/warmup_steps
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it-warmup_steps)/(max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)



optimizer = model.configure_optimizers(weight_decay = 0.1,learning_rate = 6e-4,device = device)#torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),eps=1e-8)
for step in range(max_steps):
    t0 = time.time()



    if step % 100 ==0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x,y = val_loader.next_batch()
                x,y = x.to(device), y.to(device)
                with torch.autocast(device_type=device,dtype=torch.bfloat16):
                    logits,loss = model(x,y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum,op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation_loss:{val_loss_accum.item():.4f}")

    if step % 100 ==0:
        num_return_sequences = 5
        max_length = 2048
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode("Hello, I am a language model")
        tokens = torch.tensor(tokens,dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
        x = tokens.to('cuda')

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        while x.size(1) < max_length:
            with torch.no_grad():
                logits = model(x)
                logits = logits[:,-1,:]
                probs = F.softmax(logits,dim=-1)
                topk_probs,topk_indices = torch.topk(probs,50,dim=-1)
                ix = torch.multinomial(topk_probs,1)
                xcol = torch.gather(topk_indices,-1,ix)
                x = torch.cat((x,xcol),dim=1)

        for i in range(num_return_sequences):
            tokens = x[i,:max_length].tolist()
            decoded = enc.decode(tokens)
            print(">",decoded)



    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for microstep in range(grad_accum_steps):

        x,y = train_loader.next_batch()
        x,y = x.to(device),y.to(device)
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
           logits,loss = model(x,y)
        loss = loss/grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (microstep == grad_accum_steps-1)
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum,op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1-t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed/dt
    if master_process:

        print("tokens per second: ",tokens_per_sec)
        print(f"step:{step:4d},loss:{loss_accum.item()}")
if ddp:
    destroy_process_group()
