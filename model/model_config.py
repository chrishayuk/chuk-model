from pydantic import BaseModel

class ModelConfig(BaseModel):
    name: str
    description: str
    d_model: int
    nhead: int
    num_decoder_layers: int
    dim_feedforward: int
    max_seq_length: int
    vocab_size: int = None
