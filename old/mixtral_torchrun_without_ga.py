#Based on Llama from Meta (https://github.com/meta-llama/llama/blob/main/llama/model.py) 
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from tokenizers import Tokenizer
from pathlib import Path
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers.models.prophetnet.modeling_prophetnet import ProphetNetDecoderModelOutput
import wandb
from tqdm import tqdm
from functools import partial
import tiktoken
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import bitsandbytes as bnb  # Add this import
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from torch.utils.checkpoint import checkpoint
# from dotenv import load_dotenv

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)




# import wandb
# wandb.login()


# from torch.utils.tensorboard import SummaryWriter


from datasets import load_dataset, concatenate_datasets

torch.cuda.set_device('cuda:0')



TOKEN = '...'
tinystories = True
fw = False
fw_train = None
fw_test = None

# Load training data
train_data = load_dataset(
    "roneneldan/TinyStories",
    split="train",
    # streaming=True,
    token=TOKEN
)
val_data = load_dataset(
    "roneneldan/TinyStories",
    split="validation",
    # streaming=True, 
    token=TOKEN
)




def setup(rank=None, world_size=None):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl")
    # torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    
def cleanup():
    destroy_process_group()



@dataclass
class ModelArgs:
    #Hyperparameters
    
    epochs = 4
    block_size = 256
    batch_size = 64
    embeddings_dims = 512
    attn_dropout = 0.1
    no_of_heads = 8
    dropout = 0.1
    # epochs = 100
    val_epochs = 2
    max_lr = 1e-4
    no_of_decoder_layers = 8 #IMP needs to be thoroughly calculated
    weight_decay_optim = 0.01
    beta_1 = 0.9
    beta_2 = 0.95
    clip = 1.0
    device = 'cuda'
    # no_kv_heads = 2
    vocab_size = 32768 #powers of 2 so nice!
    eps = 1e-6
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
#     dtype = 'bfloat16'
    experts=8
    top_experts=2
    use_checkpointing: bool = False
    noisy_topk: bool = True



def _save_snapshot(model, optimizer, scheduler, epoch, step):
    snapshot = {
        "MODEL_STATE": model.state_dict(),
        "OPTIMIZER_STATE": optimizer.state_dict(),
        "SCHEDULER_STATE": scheduler.state_dict() if scheduler is not None else None,  # Save scheduler state
        "EPOCHS_RUN": epoch,
        "STEP_RUN": step
    }
    torch.save(snapshot, f"/kaggle/working/snapshot_{step}.pt")
    print(f"Epoch: {epoch} | Step: {step} | Snapshot saved.")


def _load_snapshot(snapshot_path, model, optimizer, scheduler):
    snapshot = torch.load(snapshot_path)
    model.load_state_dict(snapshot["MODEL_STATE"])
    optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
    if scheduler is not None and "SCHEDULER_STATE" in snapshot and snapshot["SCHEDULER_STATE"] is not None:
        scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])  # Load scheduler state
    epoch = snapshot["EPOCHS_RUN"]
    step = snapshot["STEP_RUN"]
    print(f"Resuming from Epoch {epoch}, Step {step}")
    return epoch, step






tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token = TOKEN)
# tokenizer = PreTrainedTokenizerFast(
#     tokenizer_file=f"{MODEL_PREFIX}.model",
#     unk_token="[UNK]",
#     pad_token="[PAD]",
#     bos_token="[BOS]",
#     eos_token="[EOS]"
# )
# tokenizer.pad_token = tokenizer.eos_token
# if tokenizer.pad_token is None:
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# print("ADDED THE TOKENS: ", tokenizer.pad_token_id)
# tokenizer.bos_token = "[INST]"  
# tokenizer.eos_token = "[/INST]"  
# model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        # max_length=ModelArgs.block_size,
        padding='longest',  # Changed to dynamic padding
        truncation=True,
        return_overflowing_tokens=True,
        return_tensors='pt'
    )




def prepare_dataset(split, device, batch_size):
    print("Device is: ", device)

    def collate_fn(batch):
        # Extract text data
        texts = [item ["text"] for item in batch]

        # Set the pad token if it isn't set already
        # if tokenizer.pad_token is None:
        #     tokenizer.pad_token = tokenizer.eos_token
        # outputs = []
        # texts = []
        # for item in batch:
        #     instruction = item['prompt']
        #     # input = item['input']
        #     output = item['completion']
        #     # out = alpaca_prompt.format(instruction, output)
        #     texts.append(instruction)
        #     outputs.append(output)
        # Tokenize text data
        input_encodings = tokenizer(texts, padding='max_length', max_length=ModelArgs.block_size, truncation=True, return_tensors="pt")
        # output_encodings = tokenizer(outputs, max_length = ModelArgs.block_size, padding='max_length', truncation=True, return_tensors="pt")
        # input_encodings["labels"] = tokenizer(outputs, max_length = ModelArgs.block_size, padding='max_length', truncation=True, return_tensors="pt")
        # out = {"input": input_encodings}
        # input_encodings['input_ids'][: , input_encodings["attention_mask"] == 0] = -100
        input_encodings["labels"] = input_encodings["input_ids"].clone()  # Use `input_ids` as labels
        
        input_encodings["labels"][:, :-1] = input_encodings["input_ids"][:, 1:]  # Shift right
        input_encodings["labels"][:, -1] = tokenizer.eos_token_id  # Let the last token be end 
        # Return tokenized input tensors
        # return out
        return input_encodings

    # Create DistributedSampler for proper shuffling and partitioning across processes
    # dist_sampler = DistributedSampler(fw_train["text"], shuffle=True)

    # Create DataLoader with custom collate_fn
    # print(fw_dataset)
    dataloader = None
    if(tinystories):
        if(split == 'train'):
            data_loader = DataLoader(
            train_data,
            # generator=generator,
            batch_size=batch_size,
             
            # sampler=DistributedSampler(fw_train, shuffle=True),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=True,
            #  pin_memory=True,  # Add this
            # persistent_workers=True
        )
        elif(split == 'val'):
            data_loader = DataLoader(
            val_data,
              
            
            batch_size=batch_size,
            # sampler=DistributedSampler(fw_test, shuffle=True),
            collate_fn=collate_fn,
            drop_last=True,
            # shuffle=True, 
            # pin_memory=True,  # Add this
            # persistent_workers=True
        )
    elif(fw):
        if(split == 'train'):
            data_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            
            
            # sampler=DistributedSampler(fw_train['train'], shuffle=True),
            collate_fn=collate_fn,
            drop_last=True,
            # shuffle=True,
            # num_workers=os.cpu_count(),
            # num_workers = min(4, os.cpu_count()//2),  # Don't overallocate
            # prefetch_factor = 2,  # Balance memory/performance       
            # pin_memory=True,  # Add this
            # persistent_workers=True
    )
        elif(split == 'val'):
            data_loader = DataLoader(
            val_data,
            batch_size=batch_size,
                # generator=generator,
            # sampler=DistributedSampler(fw_train["test"]),
            collate_fn=collate_fn,
            # num_workers=os.cpu_count(),
            # num_workers = min(4, os.cpu_count()//2), # Don't overallocate
            # prefetch_factor = 2,  # Balance memory/performance
            drop_last=True,
            # shuffle=True,
            # pin_memory=True,  # Add this
            # persistent_workers=True
        )
    return data_loader





# import numpy as np
class RotaryEmbeddings(nn.Module):
    def __init__(
        self,
         device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        batch_size: int = ModelArgs.batch_size
    ):
        super().__init__()

        self.embeddings_dims = embeddings_dims
        self.block_size = block_size
        self.batch_size = batch_size
        self.theta = 0
        self.device=device

    
    def apply_rope(self, seq):
        batch_size, seq_len, embeds_dims = seq.shape
        # print(seq.shape)
        # print(self.embeddings_dims)
        # self.matrix = torch.zeros((seq_len, self.embeddings_dims, self.embeddings_dims), dtype=torch.float32,  requires_grad=False,  device = self.device)
        token_idx = torch.arange(0 , seq_len, dtype=torch.float32,  device = self.device).unsqueeze(1)
        positions = torch.arange(0 , embeds_dims, 2, dtype=torch.float32,  device = self.device).unsqueeze(0)
        # dims = torch.arange(1, self.embeddings_dims // 2,  dtype=torch.float32)
        theta = 10000 ** (-2 * (positions) / embeds_dims)
        angles = token_idx * theta
        angles = angles.expand(seq_len, -1) # because this thing needs to be applied to every sequence in the batch but with embeds dims halved
        x_reshaped = seq.view(batch_size, seq_len, embeds_dims // 2, 2)
        
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        # print(cos_angles.shape)
        # print(sin_angles.shape)
        # print(x_reshaped.shape)
        # indices = torch.arange(self.embeddings_dims,  dtype=torch.int64,  device = self.device)

        out = torch.stack([x_reshaped[..., 0]*cos_angles - (x_reshaped[...,1] * sin_angles), x_reshaped[...,1] * cos_angles + x_reshaped[..., 0] * sin_angles], dim=-1)
        out = out.view(batch_size, seq_len, embeds_dims)
        return out

    def forward(self, x):
        # print("X shape: ", x.shape)
        # print("X is: ", x)
        # B,T,C = x.shape
        # print("MATRIX:",x)
        # if(x > self.block_size or x < self.block_size):
        #     matrix = self.init_matrix(x)
        #     return matrix
        # else:
        #     matrix = self.init_matrix(self.block_size)

        #     return matrix
        # if(ModelArgs.inference):
        res = self.apply_rope(x)
        return res 
        # else:
            # return self.x_reshaped
    
class RotaryAttentionHead(nn.Module):
    def __init__(
        self,
         device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        no_of_heads: int = ModelArgs.no_of_heads,
        attn_dropout: int = ModelArgs.attn_dropout
    ):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False, dtype=torch.float32,  device = device)
        self.key = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False, dtype=torch.float32,  device = device)
        self.value = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False, dtype=torch.float32,  device = device)
        self.rope = RotaryEmbeddings(embeddings_dims=self.head_size,  device = device)
        self.dropout = nn.Dropout(p = attn_dropout)
        self.device = device
    def forward(self,x):
        # print(x.shape)
        # print("X is: ", x)
        batch, block_size, embeddings_dims = x.shape
        query = self.query(x)
        # print(query)
        key = self.key(x)
        values = self.value(x)
        # matrix = self.rotary_matrix(block_size)
        rotary_q = self.rope(query)
        rotary_k = self.rope(key)
        
        # print(matrix.shape)
        # print(query.shape)
        masked = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))
        # rotary_query = matrix @ query.permute(1,2,0) # (B,T, C,C) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
        # rotary_key = matrix @ key.permute(1,2,0)  #  (B,T, C,C  ) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
        weights = rotary_q.permute(2,0,1) @ rotary_k.permute(2,0,1).transpose(-2, -1)#(B,T,C,T) @ (B,T,C,T) = (T,C,C,T)
        weights_masked = weights.masked_fill(masked == 0, float('-inf'))
        scaled_weights = weights_masked / (torch.sqrt(torch.tensor(key.shape[-1])))
        scaled_weights = F.softmax(scaled_weights, dim=-1)
        value = scaled_weights @ values
        out = self.dropout(value)
        return out



# Text embeddings
class TextEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size = ModelArgs.vocab_size,
        embeddings_dims = ModelArgs.embeddings_dims,
        device = ModelArgs.device
    ):
        super().__init__()
        self.embeddings_table = nn.Embedding(num_embeddings = vocab_size, embedding_dim=embeddings_dims, device=device) #Just a look up table to convert the toekns_ids to some numbers
        # nn.init.normal_(self.embeddings_table.weight.data, mean=0, std=0.02)

    def forward(self, x):
        return self.embeddings_table(x)


#Layer Normalization

class LayerNormalization(nn.Module):
    def __init__(
        self,
        embeddings_dims = ModelArgs.embeddings_dims,
        device = ModelArgs.device
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embeddings_dims)

    def forward(self, x):
        return self.layer_norm(x)



class Swish(nn.Module):
    def __init__(
        self,
        block_size: int = ModelArgs.block_size,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        device = ModelArgs.device
    ):
        super().__init__()

        self.sig = torch.nn.Sigmoid()


    def forward(self, x):
        swish = x * self.sig(x)

        return swish



class SWiGLUExpertMoE(nn.Module):
    def __init__(
        self,
        block_size: int = ModelArgs.block_size,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        device = ModelArgs.device
    ):
        super().__init__()

        self.hidden_dims = int(2 * ( 4 * embeddings_dims) / 3)
        self.swish = Swish(block_size=block_size, embeddings_dims=embeddings_dims, device=device)
        self.linear_layer1 = nn.Linear(in_features=embeddings_dims, out_features=self.hidden_dims,  bias=False, device = device)
        self.linear_layer2 = nn.Linear(in_features=embeddings_dims, out_features=self.hidden_dims,  bias=False, device = device)
        self.linear_layer3 = nn.Linear(in_features=self.hidden_dims, out_features=embeddings_dims,  bias=False, device = device)




    def forward(self, x):
        swish_res = self.swish(self.linear_layer1(x))
        x_V = self.linear_layer2(x)
        res = torch.mul(swish_res, x_V)
        out = self.linear_layer3(res)
        return out


#MoE Layer

class MoeLayer(nn.Module):
    def __init__(
        self,
        dropout = ModelArgs.dropout,
        embeddings_size = ModelArgs.embeddings_dims,
        device = ModelArgs.device,
        # inner_dimensional_states: int = 3072
    ):
        super().__init__()

        self.heads = nn.ModuleList([SWiGLUExpertMoE() for _ in range(ModelArgs.experts)])
        self.gate = nn.Linear(in_features=embeddings_size, out_features=ModelArgs.experts, device=device, bias=False)
        if(ModelArgs.noisy_topk is True and ModelArgs.use_checkpointing == False):
            self.noise = nn.Linear(in_features=embeddings_size, out_features=ModelArgs.experts, device=device, bias=False)
        # self.outputs = torch.zeros((batch_size,block_size, embeddings_size), device=device) #batch size needs to be defined because we are accessing it explicitly
        self.device = device
    def forward(self, x):
        # mlp_weights_init = self.mlp.apply(weights_init)
        self.gate_out = self.gate(x) #[bz, seq, num_experts]
        if(ModelArgs.noisy_topk == True and ModelArgs.use_checkpointing == False):
            noise = self.noise(x)
            gaussian_noise = torch.normal(0, 1, size=self.gate_out.shape, device=self.device)
            noisy_router = F.softplus(noise) * gaussian_noise
            noisy_router += self.gate_out
        else:
            noisy_router = self.gate_out
        top_k_values, top_k_indices = torch.topk(noisy_router, k=ModelArgs.top_experts) #[bs, seq len, top k]
        probs = torch.nn.functional.softmax(top_k_values, dim=-1) #[bs, seq len, top k]

        out = 0

        out = torch.zeros_like(x)
        for expert_idx in range(ModelArgs.experts):
            # Create mask for current expert across all top_k positions
            expert_mask = (top_k_indices == expert_idx)
            
            # Sum probabilities for current expert
            expert_weights = (probs * expert_mask).sum(dim=-1)  # [batch, seq_len]
            
            # Get inputs where expert is used
            selected = expert_weights > 0
            if not selected.any():
                continue
                
            # Process all selected inputs through expert
            expert_out = self.heads[expert_idx](x[selected])
            
            # Weight and accumulate outputs
            out[selected] += expert_out * expert_weights[selected].unsqueeze(-1)

        return out

class AttentionHead(nn.Module):
    def __init__(
        self,
        attn_dropout = ModelArgs.attn_dropout,
        embeddings_dims = ModelArgs.embeddings_dims,
        no_of_heads = ModelArgs.no_of_heads,
        device = ModelArgs.device
    ):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,device=device, bias=False)
        self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device,bias=False)
        self.dropout = nn.Dropout(p = attn_dropout)
        self.device = device
        self.rotary= RotaryEmbeddings(embeddings_dims=self.head_size,  device = device)

    def forward(self, x):
        batch, block_size, embd_dims = x.shape
        k = self.keys(x)
        q = self.query(x)
        v = self.values(x)
        q = self.rotary(q)
        k = self.rotary(k)
        masked_table = torch.tril(torch.ones(block_size, block_size, device=self.device))
        weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
        masked_values = weights.masked_fill(masked_table[: block_size, : block_size] == 0, float('-inf'))
        weights_normalized = nn.functional.softmax(masked_values, dim=-1) #Normalize along the embeddings dimension for all the tokens
        weights_normalized = self.dropout(weights_normalized)
        out = weights_normalized @ v
        return out



# MHA




class MHA(nn.Module):
    def __init__(
        self,
        attn_dropout = ModelArgs.attn_dropout,
        embeddings_dims = ModelArgs.embeddings_dims,
        no_of_heads = ModelArgs.no_of_heads,
        device = ModelArgs.device
    ):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, device=device) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p = attn_dropout)
        self.linear = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=device, bias=False) # 12 (no of heads) * (batch_size) 64 = 768 -> gives out the text embeddings

    def forward(self, x):
        concat = torch.cat([head(x) for head in self.heads], dim=-1)
        linear_layer = self.linear(concat)
        out = self.dropout(linear_layer)
        return out



# Decoder Block

class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        attn_dropout = ModelArgs.attn_dropout,
        embeddings_dims = ModelArgs.embeddings_dims,
        no_of_heads = ModelArgs.no_of_heads,
        dropout = ModelArgs.dropout,
        vocab_size = ModelArgs.vocab_size,
        device = ModelArgs.device   
    ):
        super().__init__()

        self.mha = MHA(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, device=device)
        self.layer_norm1 = LayerNormalization(embeddings_dims=embeddings_dims, device=device)
        self.layer_norm2 = LayerNormalization(embeddings_dims=embeddings_dims, device=device)
        self.moe_block = MoeLayer(dropout=dropout, embeddings_size=embeddings_dims, device=device)

    def forward(self, x):
        # x = self.mha(x)
        # x = x + self.layer_norm1(x)
        # x = x + self.mlp_block(x)
        # out = self.layer_norm2(x)
        x = x + self.mha(self.layer_norm1(x))  #Very important step -> Layer Norm on input and then passes it to the subsequent blocks
        x = x + self.moe_block(self.layer_norm2(x)) #Very important step

        return x


# Decoder Block

class Mixtral(nn.Module):
    def __init__(
        self,
        attn_dropout = ModelArgs.attn_dropout,
        embeddings_dims = ModelArgs.embeddings_dims,
        no_of_heads = ModelArgs.no_of_heads,
        block_size = ModelArgs.block_size,
        dropout = ModelArgs.dropout,
        no_of_decoder_layers = ModelArgs.no_of_decoder_layers,
        vocab_size = ModelArgs.vocab_size,
        device = ModelArgs.device
    ):
        super().__init__()

        self.positional_embeddings = nn.Parameter(torch.randn(1, block_size, embeddings_dims, device=device), requires_grad=True) #To give positional embeddings to each token of the input text, hence num_embeddings=block_size
        torch.nn.init.kaiming_normal_(self.positional_embeddings)
        self.text_embds = TextEmbeddings(vocab_size=vocab_size, embeddings_dims=embeddings_dims, device=device)
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=vocab_size, device=device, bias=False) # Takes in logits of dimensions- embeds_dims and converts it into dimension of vocab_size (logits in range of vocab_size)
        self.layer_norm = LayerNormalization(embeddings_dims=embeddings_dims)
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, dropout=dropout, vocab_size=vocab_size, device=device) for _ in range(no_of_decoder_layers)])
        self.apply(self.kaiming_init_weights)

    def kaiming_init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Embedding):
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.text_embds(x)
        # x = x + self.positional_embeddings[: , :x.shape[1], :] #@@@Important remember
        for layer in self.decoder_layers:
            if(ModelArgs.use_checkpointing):
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        x = self.layer_norm(x)
        out = self.linear_layer(x)
        return out



# from andrej karapathy github
def topk_sampling(model, prompt, device, max_length=50, top_k=50, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated_tokens = []
    input_ids_len = len(input_ids[0])
    ModelArgs.inference=True
    for _ in range(max_length - input_ids_len):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            
            # Top-k filtering
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            
            
            # Apply temperature scaling
            # probs = probs / temperature
            
            # Sample from top-k
            next_token = torch.multinomial(top_k_probs, num_samples=1)
           
            
            # generated_tokens.append(next_token.item())
            
            xcol = torch.gather(top_k_indices, -1, next_token)
            # generated_tokens.append(xcol)
            input_ids = torch.cat([input_ids, xcol], dim=1) #1 because is it the dimension of the sequence
            
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def beam_search(model, tokenizer, prompt, beam_width=5, max_length=50, temperature=1.0):
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
    beam_scores = torch.zeros(beam_width, device=device)
    beam_sequences = input_ids.repeat(beam_width, 1)

    for _ in range(max_length):
        outputs = model(beam_sequences)
        logits = outputs[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, beam_width, dim=-1)

        # Expand beams
        beam_scores = beam_scores.unsqueeze(-1) + torch.log(top_probs)
        beam_scores = beam_scores.view(-1)
        top_indices = top_indices.view(-1)

        # Select top beams
        beam_scores, top_beams = torch.topk(beam_scores, beam_width)
        beam_sequences = torch.cat([beam_sequences[top_beams // beam_width], top_indices[top_beams].unsqueeze(-1)], dim=-1)

    # Return the best sequence
    best_sequence = beam_sequences[0]
    return tokenizer.decode(best_sequence, skip_special_tokens=True)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
# ModelArgs.device = device
model = Mixtral(attn_dropout=ModelArgs.attn_dropout, embeddings_dims=ModelArgs.embeddings_dims, no_of_heads=ModelArgs.no_of_heads, block_size=ModelArgs.block_size, dropout=ModelArgs.dropout, no_of_decoder_layers=ModelArgs.no_of_decoder_layers, vocab_size=ModelArgs.vocab_size, device=ModelArgs.device)
model = model.to(ModelArgs.device)

# Printing a summary of the architecture
# !pip install torchinfo
from torchinfo import summary
# idx, targets = get_batch('test')
idx = torch.randint(
        low=0,
        high=ModelArgs.vocab_size,
        size=(ModelArgs.batch_size, ModelArgs.block_size),
        dtype=torch.long
    )
# sample_idx = random.randint(range(len(train_dataset)))
# idx, targets = train_dataset[0]
idx = idx.to(ModelArgs.device)
# print("hre")
# targets = targets.to(ModelArgs.device)
print(summary(model=model,
        input_data=idx,
        # input_size=(ModelArgs.batch_size, ModelArgs.block_size, ModelArgs.embeddings_dims),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]))

# print("ghdgh")
def find_unused_parameters(model):
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused.append(name)
    return unused

def greedy_decode(
    model, 
    tokenizer, 
    prompt, 
    device,
    max_length=50, 
    repetition_penalty=1.2, 
    context_window=10, 
    temperature=1.0, 
    eos_token_id=None,
    
):
    # model.eval()
    # device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
    generated_tokens = []
    eos_token_id = eos_token_id or tokenizer.eos_token_id  # Use EOS token if provided
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs[:, -1, :]  # Get logits for the last token

            # Apply temperature scaling
            # if temperature != 1.0:
                # logits = logits / temperature

            # Apply repetition penalty
            # if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                # for token in set(generated_tokens[-context_window:]):  # Penalize recent tokens
                    # logits[0, token] /= repetition_penalty

            # Greedy selection
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            generated_tokens.append(next_token.item())

            # Stop if EOS token is generated
            # if next_token.item() == eos_token_id:
            #     break

            # Append the new token to the input
            input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode the generated tokens
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)



def save_to_file(step, text):
    
    with open('generations.txt', 'a') as f:
        f.write(f"------------------------------------------------Step: {step}--------------------------------------------\n\n")
        f.write(text + "\n\n")
        
    
#Train the  model


# writer = SummaryWriter(log_dir="runs/experiment")

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

# Warmup phase for 2000 steps
def warmup_fn(step):
    if step < 2000:
        return step / 2000  # LR gradually increases
    return 1.0


from torch.optim.lr_scheduler import LambdaLR

def trapezoidal_lr_scheduler(optimizer, max_lr, total_steps, warmup_steps, plateau_steps, decay_steps):
    """
    Trapezoidal learning rate scheduler:
    - Increases linearly for `warmup_steps` steps.
    - Remains constant for `plateau_steps` steps.
    - Decreases linearly for `decay_steps` steps.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        elif step < warmup_steps + plateau_steps:
            # Constant plateau
            return 1.0
        else:
            # Linear decay
            decay_step = step - (warmup_steps + plateau_steps)
            return max(0.0, float(decay_steps - decay_step) / float(max(1, decay_steps)))

    return LambdaLR(optimizer, lr_lambda)


torch.set_float32_matmul_precision('high')

scaler = torch.amp.GradScaler(enabled=(ModelArgs.dtype == 'float16'))

save_checkpoint_iter = 1000
total_iters = 256000
eval_iters = 50
eval_check = 100
warmup_iters = 700
min_lr = 0.1 * ModelArgs.max_lr
lr_decay_iters = 256000
total_batch_size = 524288
micro_batch_size = ModelArgs.batch_size
gradient_accumulation_steps = total_batch_size // (micro_batch_size * (ModelArgs.block_size * torch.cuda.device_count()))

# learning rate decay scheduler (cosine with warmup) from https://github.com/karpathy/nanoGPT/blob/master/train.py


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return ModelArgs.max_lr * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (ModelArgs.max_lr - min_lr)


def train():
    # setup()
    # device = int(os.environ["LOCAL_RANK"])
    device = 0
    torch.cuda.set_device(int(device))
    # torch.set_default_device('cuda')
    # train_dataloader = prepare_dataset(ModelArgs.batch_size)
    # rank = torch.distributed.get_rank()
    print(f"Start running DDP on rank {device}.")
    # # create model and move it to GPU with id rank
    # device_id = rank % torch.cuda.device_count()
    # CFG = ModelArgs()

    if(device == 0):

       
    
#         # Initialise run
        wandb.init(
            # entity = 'rajceo2031',
                        project = 'Mixtral-DDP-Pretrain-10-billion-tokens',
                        # config = CFG,
                        # save_code = True,
                        #group = 'ANN',
                        #job_type = 'train'
)
    print("wandb initialized")
    
    model = Mixtral(attn_dropout=ModelArgs.attn_dropout, embeddings_dims=ModelArgs.embeddings_dims, no_of_heads=ModelArgs.no_of_heads, block_size=ModelArgs.block_size, dropout=ModelArgs.dropout, no_of_decoder_layers=ModelArgs.no_of_decoder_layers, vocab_size=ModelArgs.vocab_size, device=device)
    
    # print(f"Model on device {device} is ready")
    print(f"Model on device {device} is ready")
    

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=ModelArgs.max_lr, 
        betas=(ModelArgs.beta_1, ModelArgs.beta_2),
        weight_decay=ModelArgs.weight_decay_optim,
        eps=ModelArgs.eps,
        
    )
    # model = torch.compile(model)
    model = model.to(device)
    
    
    
    def cyclical_lr(step, base_lr=min_lr, max_lr=ModelArgs.max_lr, step_size=1500):
        cycle = math.floor(1 + step / (2 * step_size))
        x = abs(step / step_size - 2 * cycle + 1)
        return base_lr + (max_lr - base_lr) * max(0, (1 - x))

# Initialize the scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cyclical_lr(step))



    
    model.eval()
    world_size = torch.cuda.device_count()
    @torch.inference_mode()
    def estimate_loss(val_loader, val_iterator, device):
        out = {}
        # train_loader = prepare_dataset('train', ModelArgs.batch_size)
        
        # val_loader_iterator = iter(val_loader)
        loader = None
        epoch_loss = None
        epoch_losses = []
        # print("Starting the eval...")
        for split in ['val']:
            print(f"Starting with {split} evaluation...")
            # losses = torch.zeros(ModelArgs.val_epochs)
            # if(split == 'train'):
            #         loader = train_loader
            # if(split == 'val'):
            #         loader = val_loader
            for step in range(eval_check):  
                try:
                    batch = next(val_iterator)
                except StopIteration:
                    val_loader_iterator = iter(val_loader)
                    batch = next(val_loader_iterator)
                
                total_loss = 0  
                # loader.sampler.set_epoch(step)
                total_batches = 0 
                # batch = next(val_loader_iterator)
                # for batch in loader:  # Loop through DataLoader batches
                idx = batch['input_ids']
                targets = batch['labels']
                idx = idx.to(device)
                targets = targets.to(device)
                # with torch.autocast(device_type=device, dtype=torch.float16):
                
                logits = model(idx)
                batch_size, block_size, embeddings_dims = logits.shape
                logits = logits.view(batch_size * block_size, embeddings_dims)
                targets = targets.view(batch_size * block_size)

                loss = F.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)

                total_loss += loss.item()
                total_batches += 1

            # Compute mean loss for this epoch
            epoch_loss = total_loss / total_batches if total_batches > 0 else 0.0
            epoch_losses.append(epoch_loss)

                # print(f"Epoch {epoch + 1}/{ModelArgs.val_epochs}: Loss = {epoch_loss:.4f}")

            # Compute mean loss across all evaluation epochs
            out[split] = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            epoch_loss = None
            epoch_losses = []

        model.train()
        return out

    # model = model.to(rank)
    model.train()
    count = 0
   
    train_dataloader = prepare_dataset('train', device, ModelArgs.batch_size)
    val_loader= prepare_dataset('val', device, ModelArgs.batch_size)
    # for step in tqdm(range(total_iters)):
    # for epoch in range(ModelArgs.epochs):
        # torch.cuda.synchronize() 
    
    # train_dataloader.sampler.set_epoch(epoch)
    
    # val_loader.sampler.set_epoch(epoch)
    print("Loaders ready both")
    epochs = ModelArgs.epochs

    # train_step_iterator = range(len(train_dataloader))
    # if device == 0:  # Only create progress bar on rank 0
    #   train_step_iterator = tqdm(train_step_iterator, desc="Training Progress", position=0, leave=True)

        # Print progress on rank 0
    train_loader_length = 0
    train_data_iterator = iter(train_dataloader)
    val_data_iterator = iter(val_loader)
    token_count = 0
    # if(device == 0):
        # train_loader_length = len(train_dataloader)
        # print("Total batches: ", train_loader_length)
    # print("Length of : ", len(train_dataloader))
    # print("Length of val: ", len(val_loader))
    # for  step, batch in enumerate(train_dataloader):
    for step in tqdm(range(total_iters)):
        # print("Dataloader things: ", batch)
        # print("Total batches: ", len(train_dataloader))
        
        
        if(device == 0):
            # if(step % 100 == 0):
        #     if(step == train_loader_length):
        #       break
                print("Step : ", step, "/", total_iters)
                # print('Total batches: ', len(train_dataloader))
                print("Total gradient accumulation steps: ", gradient_accumulation_steps)
                print("Total tokens processed: ", token_count)
                
        # all_gpus_avg_train_loss = None
        # all_gpus_avg_val_loss = None
        # every once in a while evaluate the loss on train and val sets
        if (step  % eval_iters == 0 and step != 0) or step == total_iters - 1:
            losses = estimate_loss( val_loader, val_data_iterator, 'cuda')
            # avg_train_loss = losses['train']
            avg_val_loss = losses['val']
            # print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # if device == 0:  # Only print on main process
            print(f"[GPU {device}] | Step: {step} / {total_iters} | Val Loss: {losses['val']:.4f}")
            # print(f"[GPU {device}] | Epoch {epoch}/{ModelArgs.epochs}| |Step: {step} | Train Loss: {losses['train']:.4f}")
                # print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                # Log training loss more frequently
                # Aggregate average loss across all GPUs
            # avg_train_loss = torch.Tensor([losses['train']]).to(device)
            avg_val_loss = torch.Tensor([losses['val']]).to(device)
            # torch.distributed.reduce(avg_train_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
            # torch.distributed.reduce(avg_val_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
            
            if device == 0:
                # all_gpus_avg_train_loss = avg_train_loss / world_size
                # print(f"All_GPUs_Train_losses: {all_gpus_avg_train_loss.item():.4f}")
                # all_gpus_avg_val_loss = avg_val_loss / world_size
                print(f"Val Loss: {avg_val_loss.item():.4f}")
                
            # if device == 0:
        
                # writer.add_scalar("All_GPUs_Train_losses", all_gpus_avg_train_loss.item(), global_step=step)
                # writer.add_scalar("All_GPUs_Val_losses", all_gpus_avg_val_loss.item(), global_step=step)
                # writer.add_scalar("training_step_loss", losses['train'], global_step=step)
                # writer.add_scalar("val_step_loss", losses['val'], global_step=step)
                # writer.add_scalar("GPU", device, global_step=step)
                # writer.add_scalar("Epoch", epoch, global_step=step)
                
                perplexity = torch.exp(torch.tensor(avg_val_loss.item()))  # Calculate perplexity

                if device == 0:
                    wandb.log({
                        "Val_Loss": avg_val_loss.item(),
                        "Val Perplexity": perplexity.item(),
                        "Total Tokens Processed": token_count,
                        "Step": step,
                    })
                    print(f"Step: {step} | Val Loss: {avg_val_loss.item():.4f} | Perplexity: {perplexity.item():.4f} | Tokens: {token_count}")
                
                
        
        #Loading a checkpoint
        # if(os.path.exists('snapshot.pt')):
        #    model, optimizer =  _load_snapshot(model=model, optimizer=optimizer, epoch=epoch, step=step, snapshot_path='snapshot.pt')
        
        # if(step % save_chechpoint_iter == 0 and device == 0 and step != 0):
            
        #     _save_snapshot(epoch=epoch, model=model, optimizer=optimizer, step=step)

            # When saving a checkpoint:
        if step % save_checkpoint_iter == 0 and device == 0 and step != 0:
            print(f"Saving the model checkpoint for step: {step}")
            _save_snapshot(model, optimizer, scheduler, None, step)  # Pass the scheduler here

# When loading a checkpoint (if needed):
# epoch, step = _load_snapshot('snapshot.pt', model, optimizer, scheduler)  # Pass the scheduler here
        
        accumulated_loss = 0.0
        
        
        optimizer.zero_grad(set_to_none=True)
        # for micro_step in range(gradient_accumulation_steps):
        try:
            batch = next(train_data_iterator)
        except StopIteration:
            train_data_iterator = iter(train_dataloader)
            batch = next(train_data_iterator)
        # print(batch)
        # batch = next(train_data_iterator)
        # print(batch)
        # batch = {k: v.to(self.local_rank) for k, v in batch.items()}
        idx = batch['input_ids'].to(device)
        # idx, targets = get_batch(split='train')
        # print(f"Starting the train step: {step}...")
        # for idx, targets in train_loader:
        # idx, targets = next(iter(train_loader))
        
        # print("Idx: ", idx)
        # print("Targets: ", targets)
        
        # idx = idx.to(device)
        # print("Idx: ", idx)
        # print("Targets: ", targets)
        targets = batch['labels'].to(device)
        token_count += len(idx)
        # with torch.autocast(device_type=ModelArgs.device, dtype=torch.float16):
        logits = model(idx)
        batch_size, block_size, embeddings_dims = logits.shape
        # print(logits.shape)
        # print(targets)
        logits = logits.view(batch_size*block_size, embeddings_dims)
        # print("OK")
        targets = targets.view(batch_size * block_size)
        # print("OK2")
        loss = nn.functional.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)
        
            # loss = loss / gradient_accumulation_steps #IDK why div is done here specifically? Maybe think of it in terms of a very big batch being processed and there is need for equal important of each mini batch for the overall big batch
            # accumulated_loss += loss.detach()
        loss.backward()    
        # model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1) # so that we dont synchronize the gradient everytime across the GPU devices
        # scaler.scale(loss).backward()
            # Check for unused parameters
        # del logits, targets, loss
        

        unused_params = find_unused_parameters(model)
        if unused_params:
            print(f"Unused parameters: {unused_params}")
    # break

        if(device == 0):
            # if(micro_step % 10 == 0):
        #     if(step == train_loader_length):
        #       break
                
            # print("Micro Batch : ", micro_step)
            print("Step : ", step, "/", total_iters)
            # print('Total batches: ', len(train_dataloader))
            # print("Total gradient accumulation steps: ", gradient_accumulation_steps)
            print("Total tokens processed: ", token_count)
        # count += 1
       
        lr = cyclical_lr(step)
        for params in optimizer.param_groups:
            params['lr'] = lr
            
        
        
        # Compute gradient norms before clipping
        if(ModelArgs.clip != 0.0):
            
            scaler.unscale_(optimizer) #To avoid underflow
            total_norm_before = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ModelArgs.clip)

            # Compute gradient norms after clipping
            total_norm_after = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
            )
            
            if(device  == 0 and step !=0):
                print(f"Gradient Norm Before Clipping: {total_norm_before.item():.4f}")
                print(f"Gradient Norm After Clipping: {total_norm_after.item():.4f}")
        
        # Compute gradient norms for each parameter
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.norm().item()
        #         print(f"Gradient norm for {name}: {grad_norm}")
        
        # scaler.step(optimizer)
        # scheduler.step()
        # scaler.update()
        # torch.cuda.empty_cache()
        optimizer.step()
        # new_scheduler.step()
        torch.cuda.empty_cache()
        torch.cuda.synchronize() 
        # torch.distributed.reduce(loss, dst=0, op=torch.distributed.ReduceOp.SUM)
        perplexity = torch.exp(torch.tensor(loss.item()))  # Calculate perplexity
        if(device == 0):
            wandb.log({
                    "Learning Rate": optimizer.param_groups[0]['lr'],
                    "Train_Loss": loss.item(),
                    "Train Perplexity": perplexity.item(),
                    "Total Tokens Processed": token_count,
                    "Step": step,
                    "Gradient Norm": total_norm_before.item(),
                    # "Epoch": epoch
                    
                })
      
        if device == 0 and step % 50 == 0:
            count = 1
            while(count):  # Only generate text on the main process
                # print("Generating text...")
                
    #             alpaca_prompt = '''
    
    #                 ### Instruction:
    #                 {}
    
    #                 ### Input:
    #                 {}
    
    #                 ### Response:
                
    #                 '''
                
                # prompt = alpaca_prompt.format("You are a helpful assistant.",  "Say a joke.",  "")
    #             print("Generating text")
                prompt = "Once upon a time, there was a pretty boy"
                generated_text = topk_sampling(model, prompt, max_length=ModelArgs.block_size, top_k=50, temperature=1.0, device=device)
    
        #         generated_text = greedy_decode(
        # model, 
        # tokenizer, 
        # "Once upon a time", 
        # max_length=40, 
        # repetition_penalty=1.2, 
        # context_window=10,
        # temperature=0.7,  # Lower temperature for more deterministic output
        # device=device
    # )
                # generated_text = beam_search(model, tokenizer, "Once upon a time ", beam_width=5, max_length=50, temperature=0.6)
                print(f" Step: {step} | Generated Text: {generated_text}")
            # model.train()
                save_to_file(step, generated_text)
                count -= 1
        
        # if step != 0:
        #         train_step_iterator.set_postfix({"Train loss": f"{all_gpus_avg_train_loss.item():.4f} | Val Loss : {all_gpus_avg_val_loss.item():.4f}"})
        


        # break
        if step % 5 == 0:
            torch.cuda.empty_cache()
    # Cleanup
    if device == 0:
        # writer.close()
        wandb.finish()
    cleanup()


world_size = torch.cuda.device_count()
print(f"World size: {world_size}")
train()




