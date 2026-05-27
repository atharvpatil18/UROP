import torch
import torch.nn as nn
import math

class ISLSeq2SeqLSTM(nn.Module):
    """
    Lightweight PyTorch Sequential LSTM for local temporal sign/alphabet classification.
    Incorporates ReLU activations and a dropout rate of 0.5.
    Input size: (batch_size, seq_len, 150)
    Output size: (batch_size, num_classes)
    """
    def __init__(self, input_dim: int = 150, hidden_dim: int = 256, num_layers: int = 2, num_classes: int = 26):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0.0
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, 150)
        out, (hn, cn) = self.lstm(x)
        # Use final time step hidden state for classification
        last_step = out[:, -1, :]
        activated = self.relu(last_step)
        dropped = self.dropout(activated)
        logits = self.fc(dropped)
        return logits


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for the Pose-Encoded Signformer.
    Adds absolute sequence order context to spatial coordinate features.
    """
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PoseEncodedSignformer(nn.Module):
    """
    Pose-Encoded Signformer architecture mapping 150-dimensional keypoint sequence
    directly into continuous spatial-temporal embedding layers.
    Consists of exactly 3 encoder layers and 3 decoder layers.
    """
    def __init__(
        self, 
        input_dim: int = 150, 
        d_model: int = 256, 
        nhead: int = 8, 
        num_encoder_layers: int = 3, 
        num_decoder_layers: int = 3, 
        dim_feedforward: int = 512, 
        dropout: float = 0.1,
        target_vocab_size: int = 100
    ):
        super().__init__()
        self.d_model = d_model
        
        # 1. Project 150-D coordinate features directly to d_model space
        self.source_embedding = nn.Linear(input_dim, d_model)
        
        # 2. Target token embedding (for text seq generation)
        self.target_embedding = nn.Embedding(target_vocab_size, d_model)
        
        # 3. Sinusoidal Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        
        # 4. Custom 3-Layer Encoder / 3-Layer Decoder Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_decoder_layers
        )
        
        # 5. Output classification head projecting to vocab logits
        self.fc_out = nn.Linear(d_model, target_vocab_size)
        
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generates causal mask for self-attention inside the decoder
        to prevent looking at future sequence steps.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_key_padding_mask: torch.Tensor = None, tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
          src: Coordinate tensor, shape (batch, src_seq_len, 150)
          tgt: Target token ids, shape (batch, tgt_seq_len)
        """
        # Embed continuous spatial coordinates & add positional coding
        src_emb = self.source_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        
        # Embed discrete text targets & add positional coding
        tgt_emb = self.target_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        
        # 3 Encoder layers processing continuous signs
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        # Causal mask for decoder autoregression
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        
        # 3 Decoder layers cross-attending to encoder representation
        out = self.transformer_decoder(
            tgt=tgt_emb, 
            memory=memory, 
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        logits = self.fc_out(out)
        return logits
