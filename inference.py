
from models import TransformerSTT
import re
import json
import sys

import torch
from torch.nn.utils.rnn import pad_sequence

global_scope = sys.modules[__name__]

import argparse
from glob import glob
import numpy as np

from texts import KOREAN_TOKENS, KOREAN_TABLE

from mel2samp_waveglow import Mel2SampWaveglow

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'NanumGothic'

from prepare_batch import normalize_tensor, pad_tensor_to_multiple, get_self_attention_key_padding_mask

from run import load_checkpoint

CONFIGURATION_FILE='config.json'

with open(CONFIGURATION_FILE) as f:
    data = f.read()
    json_info = json.loads(data)

    mel_config = json_info["mel_config"]
    MEL2SAMPWAVEGLOW = Mel2SampWaveglow(**mel_config)

    hp = json_info["hp"]

    for key in hp:
        setattr(global_scope, key, hp[key])

    model_parameters = json_info["mp"]

KOREAN_PATTERN = re.compile('[^ㄱ-ㅎ|ㅏ-ㅣ|가-힣| .,!?]')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-resume', metavar='-r', type=str,
                        help='resume train', default='./runs/nipa-no-mask-no-shuffle')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = TransformerSTT(**model_parameters)
    model = model.to(device)
    
    learning_rate = LR
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_step = 0

    model, optimizer, train_step = load_checkpoint(model, optimizer, 
                                                   args.resume, map_location='cpu')

    model.eval()

    files = sorted(glob('./wavs/*.wav'))

    for file in files:

        mel = MEL2SAMPWAVEGLOW.get_mel(file).T # (MB, T) -> (T, MB)
        mel = normalize_tensor(mel, -12)
        mels = [mel]
        mel_lengths = [mel.shape[0]]
        mel_tensor = pad_sequence(mels, batch_first=True, padding_value=-1).transpose(1, 2) # (B, T, MB) -> (B, MB, T)
        mel_tensor = pad_tensor_to_multiple(mel_tensor, 8)

        shrinked_mel_lengths = [int(np.ceil(mel_length / 8)) for mel_length in mel_lengths]
        mel_transformer_mask = get_self_attention_key_padding_mask(shrinked_mel_lengths) # (N, T / 8, T / 8)

        output_tensor = model((mel_tensor.to(device), 
                            mel_transformer_mask.to(device),
                            ))

        output_tensor = output_tensor.permute(1, 0, 2) # (N, S, E) => (T, N, C)

        decoded_output_text = KOREAN_TABLE.decode_jamo_prediction_tensor(output_tensor)
        decoded_output_str = KOREAN_TABLE.decode_ctc_prediction(decoded_output_text)
        decoded_output_str = decoded_output_str[0].replace('<s>', '').replace('</s>', '')
        
        print(file)
        print('>>>', decoded_output_str)

