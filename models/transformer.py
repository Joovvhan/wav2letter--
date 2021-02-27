'''
Transformer-Based Acoustic Models. 
Our transformer-based acoustic models have a small front-end:
3 (LIBRISPEECH AMs) or 6 (LIBRIVOX AM) layers of 1-D convolutions each of kernel width 3 and 
respective input and output sizes (80,Dc), (Dc/2,Dc), [(Dc/2,Dc), (Dc/2,Dc), (Dc/2,Dc),] (Dc/2,Dtr × 2), 
with Dc = 1024 or 2048. Each convolution is followed by a GLU activation function [12] and 
are striding by 2 each (for 3 consecutive layers), or every other layer (for 6 layers). 
The output of the front-end for all models is thus strided by 8 frames (80 ms). After the front-end, 
each Transformer block has 4 attention heads followed by a feedforward network (FFN) with one hidden layer and a ReLU non-linearity. 
There are two configurations of Transformer blocks: 
one 24 layer configuration (only for the LIBRISPEECH CTC AM) with dimension Dtr = 1024 for the self-attention and 4096 for the FFN, 
and one 36 layer configuration with dimension Dtr = 768 for the self-attention and 3072 for the FFN. 
Specifically, given a sequence of T vectors of dimension d, the input is represented by the matrix H0 ∈ Rd×T, 
following exactly [48]:
Zi = NORM(SELFATTENTION(Hi−1) +Hi−1), Hi = NORM(FFN(Zi) + Zi),
where Z is the output of the self-attention layer, with a skip connection, and H is the output of the FFN layer, with a skip connection. 
As is standard: our NORM is LayerNorm, and self-attention is defined as in Eq. 1, 
but with K = WKH, Q = WQH, and V = WVH. For CTC-trained models, the output of the encoder HLe
is followed by a linear layer to the output classes. For Seq2Seq models, we have an additional decoder, 
which is a stack of 6 Transformers with encoding dimension 256 and 4 attention heads. 
The probability distribution of the transcription is factorized as:

where y0 is a special symbol indicating the beginning of the transcription. For all layers (encoder and decoder – when present), 
we use dropout on the self-attention. We also use layer drop [14], dropping entire layers at the FFN level.
'''

'''
For CTC-trained models, the output of the encoder HLe is followed
by a linear layer to the output classes. 
For Seq2Seq models, we have an additional decoder, 
which is a stack of 6 Transformers with encoding dimension 256 and 4 attention heads.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from texts import KOREAN_TOKENS

NUM_TOKENS = len(KOREAN_TOKENS)

# Dc  = 2048
# Dc_2 = int(Dc / 2)
# Dtr = 1024
# NUM_HEAD = 4
# # TRNS_NUM_ENC = 12
# # TRNS_NUM_DEC = 12

# # TRNS_NUM_ENC = 3
# # TRNS_NUM_DEC = 3

# # TRNS_NUM_ENC = 12
# # TRNS_NUM_DEC = 12

# TRNS_NUM_ENC = 9
# TRNS_NUM_DEC = 9

# # TRNS_NUM_ENC = 1
# # TRNS_NUM_DEC = 1

# TRNS_HDN_FFN = 4096

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, input_tensor):
        print(input_tensor.shape)
        return input_tensor   

# class SelfAttention(nn.Module):

#     '''
#     src: (S, N, E)
#     tgt: (T, N, E)
#     '''

#     def __init__(self):
#         super(SelfAttention, self).__init__()
#         self.transformer = nn.Transformer(Dtr, nhead=NUM_HEAD, 
#                                           num_encoder_layers=TRNS_NUM_ENC,
#                                           num_decoder_layers=TRNS_NUM_DEC,
#                                           dim_feedforward=TRNS_FFN)
#         '''
#         def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
#             num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
#             activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None) -> None:
#         forward(src: torch.Tensor, tgt: torch.Tensor, src_mask: Optional[torch.Tensor] = None, 
#             tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None, 
#             src_key_padding_mask: Optional[torch.Tensor] = None, 
#             tgt_key_padding_mask: Optional[torch.Tensor] = None, 
#             memory_key_padding_mask: Optional[torch.Tensor] = None) → torch.Tensor
#         '''

#     def forward(self, tensor, src_mask=None, tgt_mask=None):
#         tensor  =  self.transformer(tensor, 
#                                     tensor,
#                                     # src_mask=src_mask, 
#                                     # tgt_mask=tgt_mask,
#                                     )
        
#         return tensor

def build_positional_encoding(d_model, max_pos=1000):

    PE = torch.zeros([max_pos, d_model], requires_grad=False)

    denominator = 1000
    # denominator = 1000

    x  = [[pos / denominator ** (2 * i / d_model) for i in range(int(d_model/2))] for pos in range(max_pos)]
    x_tensor = torch.sin(torch.tensor(x))
    # print(x_tensor.shape)

    y  = [[pos / denominator ** (2 * i / d_model) for i in range(int(d_model/2))] for pos in range(max_pos)]
    y_tensor = torch.cos(torch.tensor(y))
    # print(y_tensor.shape)

    PE[:, ::2] = x_tensor
    PE[:, 1::2] = y_tensor

    return PE.unsqueeze(1) # (L, 1, H)

class TransformerSTT(nn.Module):
    def __init__(self, Dc=2048, Dtr=1024, NUM_HEAD=4, 
                 TRNS_NUM_ENC=9, TRNS_NUM_DEC=9, TRNS_HDN_FFN=4096):
        super(TransformerSTT, self).__init__()

        Dc_2 = int(Dc / 2)

        # self.front_end_layers = nn.DataParallel(nn.Sequential(
        #     nn.Conv1d(80, Dc, kernel_size=3, padding=1),
        #     nn.GLU(dim=1),
        #     nn.Conv1d(Dc_2, Dc, kernel_size=3, stride=2, padding=1),
        #     nn.GLU(dim=1),
        #     nn.Conv1d(Dc_2, Dc, kernel_size=3, padding=1),
        #     nn.GLU(dim=1),
        #     nn.Conv1d(Dc_2, Dc, kernel_size=3, stride=2, padding=1),
        #     nn.GLU(dim=1),
        #     nn.Conv1d(Dc_2, Dc, kernel_size=3, padding=1),
        #     nn.GLU(dim=1),
        #     nn.Conv1d(Dc_2, 2 * Dtr, kernel_size=3, stride=2, padding=1),
        #     nn.GLU(dim=1),
        # ))

        self.front_end_layers = nn.Sequential(
            nn.Conv1d(80, Dc, kernel_size=3, padding=1),
            nn.GLU(dim=1),
            nn.Conv1d(Dc_2, Dc, kernel_size=3, stride=2, padding=1),
            nn.GLU(dim=1),
            nn.Conv1d(Dc_2, Dc, kernel_size=3, padding=1),
            nn.GLU(dim=1),
            nn.Conv1d(Dc_2, Dc, kernel_size=3, stride=2, padding=1),
            nn.GLU(dim=1),
            nn.Conv1d(Dc_2, Dc, kernel_size=3, padding=1),
            nn.GLU(dim=1),
            nn.Conv1d(Dc_2, 2 * Dtr, kernel_size=3, stride=2, padding=1),
            nn.GLU(dim=1),
        )

        # self.transformer = nn.DataParallel(nn.Transformer(Dtr, nhead=NUM_HEAD, 
        #                                   num_encoder_layers=TRNS_NUM_ENC,
        #                                   num_decoder_layers=TRNS_NUM_DEC,
        #                                   dim_feedforward=TRNS_HDN_FFN))

        self.transformer = nn.Transformer(Dtr, nhead=NUM_HEAD, 
                                          num_encoder_layers=TRNS_NUM_ENC,
                                          num_decoder_layers=TRNS_NUM_DEC,
                                          dim_feedforward=TRNS_HDN_FFN)

        self.ctc_linear = nn.Linear(Dtr, NUM_TOKENS)
        # self.seq2seq_decoder = nn.ModuleList([nn.Transformer(Dtr, nhead=NUM_HEAD, 
        #                                         num_encoder_layers=1,
        #                                         num_decoder_layers=1,
        #                                         dim_feedforward=256) for i in range(6)
        #                                      ])
        #     src: (S, N, E)
        #     tgt: (T, N, E)

        self.pe = nn.Parameter(build_positional_encoding(Dtr, 1000)) # (L, 1, H)


    def forward(self, inputs):

        input_tensor, input_mask = inputs

        # print(f'input_tensor: {input_tensor.shape}')
        # print(f'input_mask: {input_mask.shape}')
        # print(f'jamo_code_tensor: {jamo_code_tensor.shape}')
        # print(f'mel_lengths: {mel_lengths}')
        # print(f'jamo_lengths: {jamo_lengths}')

        tensor = self.front_end_layers(input_tensor) # (N, C, L) => (N, C, L)
        tensor = tensor.permute(2, 0, 1) # (N, C, L) => (S[L], N, E[C])
        # print(tensor.shape, input_mask.shape)

        L, B, H = tensor.shape

        # Apply positional encoding here 
        tensor = tensor + self.pe[:L, :, :] # (L, 1, H)

        tensor = self.transformer(tensor, tensor,
                                #   src_mask=input_mask, 
                                #   tgt_mask=input_mask,
                                src_key_padding_mask = input_mask,
                                tgt_key_padding_mask = input_mask,
                                memory_key_padding_mask = input_mask,
                                )
        tensor = tensor.permute(1, 0, 2) # (S, N, E) => (N, S, E)
        tensor = self.ctc_linear(tensor) # (N, S, E) => (N, S, E)
        output_tensor = F.log_softmax(tensor, dim=-1)
        # output_tensor = tensor.permute(1, 0, 2) # (N, S, E) => (T, N, C)

        # print(f'output_tensor: {output_tensor.shape}')

        return output_tensor
        
