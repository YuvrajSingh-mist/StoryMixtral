import torch
from config import ModelArgs
from model import Mixtral
from torchinfo import summary

# Create model instance
model = Mixtral(
    attn_dropout=ModelArgs.attn_dropout, 
    embeddings_dims=ModelArgs.embeddings_dims, 
    no_of_heads=ModelArgs.no_of_heads, 
    block_size=ModelArgs.block_size, 
    dropout=ModelArgs.dropout, 
    no_of_decoder_layers=ModelArgs.no_of_decoder_layers, 
    vocab_size=ModelArgs.vocab_size, 
    device=ModelArgs.device
)
model = model.to(ModelArgs.device)

# Create sample input
idx = torch.randint(
    low=0,
    high=ModelArgs.vocab_size,
    size=(ModelArgs.batch_size, ModelArgs.block_size),
    dtype=torch.long
)
idx = idx.to(ModelArgs.device)

# Print model summary
print(summary(
    model=model,
    input_data=idx,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"]
))
