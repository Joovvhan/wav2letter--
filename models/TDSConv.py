'''
Time-Depth Separable (TDS) Convolution Acoustic Models. 
We extend the TDS block [20] (which is composed of one 2-D convolution layer and two fully-connected layers with ReLU, 
LayerNorm and residual connections in between), by increasing the number of channels in the feature maps 
spanning the two internal fully-connected layers by a factor F > 1, so as to increase model capacity. 
Following [20], 3 sub-sampling layers, i.e. 1-D convolution layers with stride 2, are adopted to ensure 
an optimal context size for the encoder. For training with only labeled data, 
we have three groups of TDS blocks with F = 3 after each sub-sampling layers. 
There are 5, 6, and 10 blocks in each group, containing 10, 14, and 18 channels, respectively. 
To increase model capacity for unlabeled data, the three groups of TDS blocks, having fewer 4, 5, and 6 blocks 
and F = 2 in each, are equipped with much larger 16, 32, and 48 channels. All convolutions in both TDS 
and sub-sampling layers have kernel shapes of 21 × 1. 
Identical encoder architectures are shared between CTC and Seq2Seq.
Our Seq2Seq self-attention decoder performs R rounds of attention through the same N-layers of RNN-GRU 
each with a hidden unit size of 512 in conjunction with the same efficient key-value attention as in [20, 48]
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from texts import KOREAN_TOKENS

NUM_TOKENS = len(KOREAN_TOKENS)

NUM_HEAD = 1
# TRNS_NUM_ENC = 12
# TRNS_NUM_DEC = 12

'''
For training with only labeled data, we have three groups of TDS blocks with F = 3 after each sub-sampling layers. 
There are 5, 6, and 10 blocks in each group, containing 10, 14, and 18 channels, respectively.
'''

NUM_TDS_BLOCKS = [5, 6, 10]
NUM_TDS_HIDDEN  = [10, 14, 18]

CONV_KERNEL_SIZE = 21

MEL_NUM = 80

# TRNS_NUM_ENC = 1
# TRNS_NUM_DEC = 1

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, input_tensor):
        print(input_tensor.shape)
        return input_tensor   

'''
Following [20], 3 sub-sampling layers, i.e. 1-D convolution layers with stride 2, 
are adopted to ensure an optimal context size for the encoder.
'''
class SubSamplingLayer(nn.Module):
    def __init__(self, num_hidden):
        super(SubSamplingLayer, self).__init__()
        self.conv2d = nn.Conv2d(num_hidden, 
                                num_hidden, 
                                kernel_size=(1, CONV_KERNEL_SIZE), 
                                padding=(0, int(CONV_KERNEL_SIZE/2)))

    def forward(self, input_tensor):

        return

class TDSBlock(nn.Module):
    def __init__(self, num_hidden):
        super(TDSBlock, self).__init__()
        self.conv2d = nn.Conv2d(num_hidden, 
                                num_hidden, 
                                kernel_size=(1, CONV_KERNEL_SIZE), 
                                padding=(0, int(CONV_KERNEL_SIZE/2)))
        
        self.tds_layer_norm = nn.LayerNorm(num_hidden, 
                                           elementwise_affine=False)

        self.fc_layer_1 = nn.Conv2d(MEL_NUM * num_hidden, 
                                    MEL_NUM * num_hidden, 
                                    kernel_size=(1, 1))

        self.fc_layer_2 = nn.Conv2d(MEL_NUM * num_hidden, 
                                    MEL_NUM * num_hidden, 
                                    kernel_size=(1, 1))

        self.fc_layer_norm = nn.LayerNorm(MEL_NUM * num_hidden, 
                                          elementwise_affine=False)

    def forward(self, input_tensor):

        tensor = self.conv2d(input_tensor) # (B, H, M, T)
        tensor = F.relu(tensor)
        tensor = tensor + input_tensor
        tensor = tensor.permute(0, 2, 3, 1) # (B, H, M, T) => (B, M, T, H)
        tensor = self.tds_layer_norm(tensor)
        tensor = tensor.permute(0, 1, 3, 2) # (B, M, T, H) => (B, M, H, T)

        B, M, H, T = tensor.shape
        tensor_ = torch.reshape(tensor, (B, M*H, 1, T)) # (B, M*H, 1, T)
        tensor = self.fc_layer_1(tensor_)
        tensor = F.relu(tensor)
        tensor = self.fc_layer_2(tensor)
        tensor = tensor + tensor_ # (B, M*H, 1, T)
        tensor = tensor.permute(0, 3, 2, 1) # (B, M*H, 1, T) => (B, T, 1, M*H)
        tensor = self.fc_layer_norm(tensor)
        tensor = torch.reshape(tensor, (B, T, M, H)) # (B, T, M, H)
        tensor = tensor.permute(0, 3, 2, 1) # (B, T, M, H) => (B, H, M, T)

        return tensor # (B, H, M, T)

'''
All convolutions in both TDS and sub-sampling layers have kernel shapes of 21 × 1.
For training with only labeled data, we have three groups of TDS blocks with F = 3 after each sub-sampling layers. 
There are 5, 6, and 10 blocks in each group, containing 10, 14, and 18 channels, respectively
'''

'''
Our Seq2Seq self-attention decoder performs R rounds of attention through the 
same N-layers of RNN-GRU each with a hidden unit size of 512 in conjunction 
with the same efficient key-value attention as in [20, 48]
where [K,V] is 512-dimensional encoder activation and Qr
t is the query vector at time t
in round r, generated by the GRU g(·). The initial Q0
t is a 512-dimensional token embedding, and the final QR
t is linearly projected to output classes for token classification. 
In our experiments, N and R are both set to either 2 or 3 based on 
validation performance. 
We use dropout in all TDS blocks and GRUs to prevent overfitting
'''

class TDSConvSTT(nn.Module):
    def __init__(self):
        super(TDSConvSTT, self).__init__()

        self.front_end_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(21, 1), padding=(10, 0))
        )

        # self.transformer = nn.DataParallel(nn.Transformer(Dtr, nhead=NUM_HEAD, 
        #                                   num_encoder_layers=TRNS_NUM_ENC,
        #                                   num_decoder_layers=TRNS_NUM_DEC,
        #                                   dim_feedforward=TRNS_HDN_FFN))

        # self.transformer = nn.Transformer(Dtr, nhead=NUM_HEAD, 
        #                                   num_encoder_layers=TRNS_NUM_ENC,
        #                                   num_decoder_layers=TRNS_NUM_DEC,
        #                                   dim_feedforward=TRNS_HDN_FFN)

        # self.ctc_linear = nn.Linear(Dtr, NUM_TOKENS)
        # self.seq2seq_decoder = nn.ModuleList([nn.Transformer(Dtr, nhead=NUM_HEAD, 
        #                                         num_encoder_layers=1,
        #                                         num_decoder_layers=1,
        #                                         dim_feedforward=256) for i in range(6)
        #                                      ])
        #     src: (S, N, E)
        #     tgt: (T, N, E)

    def forward(self, inputs):

        input_tensor, input_mask = inputs

        # print(f'input_tensor: {input_tensor.shape}')
        # print(f'input_mask: {input_mask.shape}')
        # print(f'jamo_code_tensor: {jamo_code_tensor.shape}')
        # print(f'mel_lengths: {mel_lengths}')
        # print(f'jamo_lengths: {jamo_lengths}')

        output_tensor = self.front_end_layers(input_tensor) # (N, C, L) => (N, C, L)
        # tensor = tensor.permute(2, 0, 1) # (N, C, L) => (S[L], N, E[C])
        # # print(tensor.shape, input_mask.shape)
        # tensor = self.transformer(tensor, tensor,
        #                         #   src_mask=input_mask, 
        #                         #   tgt_mask=input_mask,
        #                         src_key_padding_mask = input_mask,
        #                         tgt_key_padding_mask = input_mask,
        #                         memory_key_padding_mask = input_mask,
        #                         )
        # tensor = tensor.permute(1, 0, 2) # (S, N, E) => (N, S, E)
        # tensor = self.ctc_linear(tensor) # (N, S, E) => (N, S, E)
        # output_tensor = F.log_softmax(tensor, dim=-1)
        # output_tensor = tensor.permute(1, 0, 2) # (N, S, E) => (T, N, C)

        # print(f'output_tensor: {output_tensor.shape}')

        return output_tensor
        