import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, max_seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(max_seq_length, d_model)
        self.d_model = d_model
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_decoder_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        
        # Handling positional encoding for batches
        position_ids = torch.arange(0, src.size(1), dtype=torch.long, device=src.device).unsqueeze(0).expand(src.size(0), -1)
        src = src + self.pos_encoder(position_ids)
        
        # Define and compute tgt_mask
        tgt_mask = torch.triu(torch.ones(src.size(1), src.size(1), device=src.device), diagonal=1).bool()
        
        # Print the shapes of key tensors for debugging
        #print("src shape:", src.shape)
        #print("tgt_mask shape:", tgt_mask.shape)
        #print("Pos Encoder Shape:", self.pos_encoder.weight.shape)

        dummy_memory = torch.zeros_like(src)
        output = self.transformer_decoder(src, dummy_memory, tgt_mask=tgt_mask)
        output = self.out(output)
        return output

if __name__ == "__main__":
    # Example Model Initialization
    vocab_size = 20
    d_model = 512
    nhead = 8
    num_decoder_layers = 3
    dim_feedforward = 2048
    max_seq_length = 20

    # instatiate the model
    model = TransformerModel(vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, max_seq_length)

    # print the model
    print(model)
