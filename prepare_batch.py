
from models import TransformerSTT

import csv 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import random
import re
import json
import math
import shutil

import scipy.io.wavfile as wavfile

import torch

import numpy as np
import os

import sys

global_scope = sys.modules[__name__]

import argparse
from glob import glob 

from texts import KOREAN_TOKENS, KOREAN_TABLE

from mel2samp_waveglow import Mel2SampWaveglow

from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'NanumGothic'

import jamotools

import torch.nn as nn

from tensorboardX import SummaryWriter

CONFIGURATION_FILE='config.json'

with open(CONFIGURATION_FILE) as f:
    data = f.read()
    json_info = json.loads(data)

    mel_config = json_info["mel_config"]
    MEL2SAMPWAVEGLOW = Mel2SampWaveglow(**mel_config)

    hp = json_info["hp"]

    for key in hp:
        setattr(global_scope, key, hp[key])
        print(f'{key} == {hp[key]}')

    model_parameters = json_info["mp"]

# TRAIN_METADATA_FILE = 'metadata_train_clean.csv'
# TEST_METADATA_FILE = 'metadata_test_clean.csv'

TRAIN_METADATA_FILTERED_FILE = 'metadata_train_clean_filtered.csv'
TEST_METADATA_FILTERED_FILE = 'metadata_test_clean_filtered.csv'

LEN_TRAIN = 736153
LEN_TEST = 14890

# SR = 22050

KOREAN_PATTERN = re.compile('[^ㄱ-ㅎ|ㅏ-ㅣ|가-힣| .,!?]')

# NUM_WORKERS = 4
# # BATCH_SIZE = 8
# BATCH_SIZE = 12
# # BATCH_SIZE = 16
# # BATCH_SIZE = 32
# # BATCH_SIZE = 24
# # BATCH_SIZE = 48
# # BATCH_SIZE = 64

# MEL_MIN = -12

# NUM_EPOCH = 22
# NUM_EPOCH = 115
# NUM_EPOCH = 3 * 24 * 2

# # LR = 1e-5
# # LR = 5e-5
# # LR = 3e-5
# LR = 1e-5

# MASKING_RATIO = 0.1

# APPLY_SPECAUG = False

# APPLY_T_SHIFT = True

# CHECKPOINT_STEPS = 300000

class SpeakerTable():

    def __init__(self, speakers):
        self.speakers  = sorted(speakers)
        self.speakers_dict = {speaker: i for i, speaker in enumerate(self.speakers)}

    def __get__(self, input_code):
        if isinstance(input_code, str):
            return self.speakers_dict[input_code]
        elif isinstance(input_code, int):
            return self.speakers[input_code]
        else:
            assert False, f'Wrong input code type for SpeakerTable {input_code}'

    def __len__(self):
        return len(self.speakers_dict)
    
    # def __repr__(self):
    #     return str(self.speakers_dict)

    def speaker_name_to_code(self, speakers):
        return [self.speakers_dict[s] for s in speakers]

    def code_to_speaker_name(self, speaker_code):
        return [self.speakers[i] for i in speaker_code]

def load_metadata(meta_file_name, num_meta = 0):
    
    print(f'Loading {meta_file_name}')
    data_pairs = list()
    with open(meta_file_name, 'r') as file:
        csv_reader = csv.reader(file)
        for line in tqdm(csv_reader, total=num_meta):
            data_pairs.append(line)
    print(random.choice(data_pairs))
    
    return data_pairs

def korean_script_sanity_check(data_pairs):

    invalid_num_list = list()

    for i, pair in tqdm(enumerate(data_pairs)):
        
        invalid_letters = KOREAN_PATTERN.findall(pair[1])
        
        if len(invalid_letters) > 0:
            # print(pair)
            # print(invalid_letters)
            invalid_num_list.append(i)
    
    for i in reversed(invalid_num_list):
        pop_data = data_pairs.pop(i)
        print(f"Removed {pop_data}")
    
    return data_pairs

# def get_self_attention_mask(valid_lengths):

#     B = len (valid_lengths)
#     max_len = max(valid_lengths)

#     self_attention_mask = torch.ones(B, max_len, max_len) # (N, T / 8, T / 8)

#     for i, length in enumerate(valid_lengths):
#         self_attention_mask[i, :length, :length] = torch.zeros(length, length)
    
#     return self_attention_mask.bool()

def load_checkpoint(model, optimizer, path):

    checkpoint_files = sorted(glob(os.path.join(path, '*.pt')))
    assert len(checkpoint_files) > 0, f'No checkpoint inside {path}'

    checkpoint_file = checkpoint_files[-1]

    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    return model, optimizer, step

def save_checkpoint(model, optimizer, step, path, keep_last_only=False):
    checkpoint_name = os.path.join(path, f'checkpoint_{step:07d}.pt')

    if keep_last_only:
        checkpoint_name = os.path.join(path, 'checkpoint.pt')

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            }, checkpoint_name)
    return

def get_self_attention_key_padding_mask(valid_lengths):

    B = len (valid_lengths)
    max_len = max(valid_lengths)

    self_attention_mask = torch.ones(B, max_len) # (N, T / 8)

    for i, length in enumerate(valid_lengths):
        self_attention_mask[i, :length] = torch.zeros(length)
    
    return self_attention_mask.bool()

def random_masking(tensor, masking_ratio=0.1):

    mask = torch.rand(tensor.shape) > masking_ratio

    return torch.mul(tensor, mask)

def mel_random_masking(tensor, masking_ratio=0.1, mel_min=-12):

    mask = torch.rand(tensor.shape) > masking_ratio

    masked_tensor = torch.mul(tensor, mask)

    masked_tensor += ~mask * mel_min

    return masked_tensor

def apply_t_shift(tensor, mel_min=-12, T=10):
    
    _, MF = tensor.shape
    
    t = torch.randint(0, T, [1])
    
    shift_tensor = torch.ones(t, MF) * mel_min

    tensor = torch.cat((shift_tensor, tensor), axis=0)
    
    return tensor

def spec_augment(tensor, mel_min=-12, T=10, F=8):

    '''
    Frequency masking is applied so that 
    f consecutive mel frequency channels [f0, f0 + f) are masked,
    where f is first chosen from a uniform distribution 
    from 0 to the frequency mask parameter F, and f0 is chosen 
    from [0, ν − f). ν is the number of mel frequency channels.
    Time masking is applied so that t consecutive time steps [t0, t0 + t) 
    are masked, where t is first chosen from a uniform distribution 
    from 0 to the time mask parameter T, and t0 is chosen from [0, τ − t).
    '''

    MT, MF = tensor.shape
    
    t0 = torch.randint(0, MT - T, [1])
    f0 = torch.randint(0, MF - F, [1])
    
    t = torch.randint(0, T, [1])
    f = torch.randint(0, F, [1])
    
    tensor[t0:t0+t, :] = mel_min
    tensor[:, f0:f0+f] = mel_min
    
    return tensor

def normalize_tensor(tensor, min_v=-12, max_v=0):
    center_v = (max_v - min_v) / 2
    tensor = tensor / center_v  + 1
    return tensor
    
def collate_function(pairs):

    mels = list()
    tags = list()
    jamo_tokens = list()
    mel_lengths = list()
    jamo_lengths = list()

    B = len(pairs)

    for pair in pairs:
        # (wav_file, clean_script, clean_jamos, tag, len(clean_script), len(clean_jamos), wav_file_dur)
        wav_file, _, clean_jamos, tag, _, _, _ =  pair
        jamo_token = torch.tensor(KOREAN_TABLE.jamo_to_jamo_code(clean_jamos, append_specials=True))
        jamo_tokens.append(jamo_token)
        tags.append(tag)
        npy_file = wav_file.replace('.wav', '.npy')
        if not os.path.isfile(npy_file) or not LOAD_MEL:
            mel = MEL2SAMPWAVEGLOW.get_mel(wav_file).T # (MB, T) -> (T, MB)
            if LOAD_MEL:
                np.save(npy_file, mel.numpy()) # (T, MB)
        else:
            mel = torch.tensor(np.load(npy_file))

        if APPLY_T_SHIFT:
            mel = apply_t_shift(mel, MEL_MIN)
        mel = mel_random_masking(mel, MASKING_RATIO, MEL_MIN)
        if APPLY_SPECAUG:
            mel = spec_augment(mel, MEL_MIN)
        mel = normalize_tensor(mel, MEL_MIN)
        mels.append(mel) 
        # print(len(jamo_token))
        jamo_lengths.append(len(jamo_token))
        mel_lengths.append(mel.shape[0])

    jamo_tensor = pad_sequence(jamo_tokens, batch_first=True, padding_value=0) # (B, S)
    # mel_tensor = pad_sequence(mels, batch_first=True, padding_value=MEL_MIN).transpose(1, 2) # (B, T, MB) -> (B, MB, T)
    mel_tensor = pad_sequence(mels, batch_first=True, padding_value=-1).transpose(1, 2) # (B, T, MB) -> (B, MB, T)
    mel_tensor = pad_tensor_to_multiple(mel_tensor, 8)

    shrinked_mel_lengths = [int(np.ceil(mel_length / 8)) for mel_length in mel_lengths]
    mel_lengths = torch.tensor(mel_lengths)
    jamo_lengths = torch.tensor(jamo_lengths)

    mel_transformer_mask = get_self_attention_key_padding_mask(shrinked_mel_lengths) # (N, T / 8, T / 8)

    return mel_tensor, jamo_tensor, mel_lengths, jamo_lengths, mel_transformer_mask, tags
    # (B, MB, T), (B, S), (N), (N), (N, T / 8, T / 8), (N)

def plot_mel_spectrograms(mel_tensor, keyword=''):

    B, M, T = mel_tensor.shape

    num_x = int(np.sqrt(B))
    num_y = int(B / num_x)

    fig, axes = plt.subplots(num_x, num_y, sharex=True, sharey=True, figsize=(24, 8), dpi=300)
    axes = axes.flatten()

    for i in range(B):
        im = axes[i].imshow(mel_tensor[i, :, :], origin='lower', aspect='auto')

    plt.tight_layout()

    fig.subplots_adjust(right=0.94)
    cbar_ax = fig.add_axes([0.96, 0.05, 0.02, 0.9])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.savefig(f'mel_sample_{keyword}.png')
    plt.close()

    return

def get_next_multiple(input_length, divider):
    return divider * math.ceil(input_length / divider)

def get_padding_length(input_length, divider):
    return get_next_multiple(input_length, divider) - input_length

def pad_tensor_to_multiple(tensor, divider):
    B, M, T = tensor.shape
    pad_len = get_padding_length(T, divider)
    tensor = torch.cat((tensor, torch.full((B, M, pad_len), fill_value=MEL_MIN)), dim=2)
    return tensor

def mel_tensor_to_plt_image(tensor, titles, step):
    
    B, H, L = tensor.shape

    x = 4
    y = int(np.ceil(B / x))

    fig, axes = plt.subplots(y, x, sharey=True, figsize=(36, 12))
    fig.suptitle(f'Mel-spectrogram from Step #{step:07d}', fontsize=24, y=0.95)
    axes = axes.flatten()
    for i in range(B):
        im = axes[i].imshow(tensor[i, :, :], origin='lower', aspect='auto')
        im.set_clim(-1, 1)
        axes[i].axes.xaxis.set_visible(False)
        axes[i].axes.yaxis.set_visible(False)
        title = jamotools.join_jamos(titles[i].replace('<s>', '').replace('</s>', ''))
        axes[i].set_title(title)
    fig.colorbar(im, ax=axes, location='right')
    fig.canvas.draw()

    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    image_array = np.swapaxes(image_array, 0, 2)
    image_array = np.swapaxes(image_array, 1, 2)

    plt.close()

    return image_array

def resume_training(resume, model, optimizer, step):
    if resume is not None:
        if resume == 'latest':
            logging_folders = sorted(glob('runs/*'))
            assert len(logging_folders) > 0, f'No folder exists inside ./runs'
            logging_path = logging_folders[-1]
        else:
            logging_path = os.path.join('runs', args.resume)

        assert os.path.isdir(logging_path), f'Invalid logging path {logging_path}'
        summary_writer = SummaryWriter(logging_path)
        model, optimizer, step = load_checkpoint(model, optimizer, summary_writer.logdir)
    else:
        summary_writer = SummaryWriter()
        shutil.copy('config.json', summary_writer.logdir)

    return model, optimizer, step, summary_writer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-resume', metavar='-r', type=str,
                        help='resume train', default=None)

    args = parser.parse_args()

    train_pairs = load_metadata(TRAIN_METADATA_FILTERED_FILE)
    test_pairs = load_metadata(TEST_METADATA_FILTERED_FILE)

    # (wav_file, clean_script, clean_jamos, tag, len(clean_script), len(clean_jamos), wav_file_dur)
    speaker_table = SpeakerTable(set([pair[3] for pair in train_pairs] + [pair[3] for pair in test_pairs]))

    print(len(speaker_table))

    def is_valid(string):
        # if string == 'kss':
        #     return True
        if string in ['prosem_f', 'prosem_m', 'kss']:
            return True
        elif 'acriil' in string or 'clova' in string:
            return True
        else:
            return False
        # elif 'acriil' in string or 'clova' in string:
        #     return True
        # else:
        #     return False

    print(len(train_pairs), len(test_pairs))

    train_pairs = list(filter(lambda x: is_valid(x[3]), train_pairs))
    test_pairs = list(filter(lambda x: is_valid(x[3]), test_pairs))

    print('>>>', len(train_pairs), len(test_pairs))

    dataset_train = DataLoader(train_pairs, batch_size=BATCH_SIZE, 
                               shuffle=True, num_workers=NUM_WORKERS,
                               collate_fn=collate_function)
    dataset_test = DataLoader(test_pairs, batch_size=BATCH_SIZE, 
                              shuffle=False, num_workers=NUM_WORKERS,
                              collate_fn=collate_function,
                              drop_last=True)
    
    '''
    DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, , collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
    '''

    print(KOREAN_TOKENS)

    cuda = torch.device('cuda')

    model = TransformerSTT(**model_parameters)
    model = nn.DataParallel(model)
    model = model.cuda()
    # print(str(model))
    learning_rate = LR
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_criterion = nn.CTCLoss(zero_infinity=True)
    train_step = 0

    model, optimizer, train_step, writer = resume_training(args.resume, 
                                                           model, 
                                                           optimizer, 
                                                           train_step)

    loss_list = list()
    wer_list = list()

    for epoch in range(NUM_EPOCH):
        model.train()
        for data in tqdm(dataset_train):
            # print(data)
            mel_tensor, jamo_code_tensor, mel_lengths, jamo_lengths, mel_transformer_mask, speakers = data

            # print(jamo_code_tensor.shape)
            decoded_result = KOREAN_TABLE.decode_jamo_code_tensor(jamo_code_tensor, no_pad=True)
            # print(decoded_result)

            speaker_code = speaker_table.speaker_name_to_code(speakers)

            _speakers = speaker_table.code_to_speaker_name(speaker_code)

            output_tensor = model((mel_tensor.to(cuda), 
                                mel_transformer_mask.to(cuda),
                                ))

            output_tensor = output_tensor.permute(1, 0, 2) # (N, S, E) => (T, N, C)

            loss = loss_criterion(output_tensor, 
                                jamo_code_tensor.to(cuda), 
                                (mel_lengths / 8).to(cuda), 
                                jamo_lengths.to(cuda))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(f'{mel_tensor.shape} => {output_tensor.shape}')
            # print(f'Loss = {loss.item()}')

            decoded_input_text = KOREAN_TABLE.decode_jamo_code_tensor(jamo_code_tensor)
            decoded_input_text = KOREAN_TABLE.decode_ctc_prediction(decoded_input_text)
            decoded_output_text = KOREAN_TABLE.decode_jamo_prediction_tensor(output_tensor)
            decoded_output_str = KOREAN_TABLE.decode_ctc_prediction(decoded_output_text)

            wer = KOREAN_TABLE.caculate_wer(decoded_input_text, decoded_output_str)
            wer_list.append(wer)
            loss_list.append(loss.item())
            train_step += 1

            if len(loss_list) >= LOGGING_STEPS:
                writer.add_scalar('ctc_loss/train', np.mean(loss_list), train_step)
                decoded_pairs =  [f'** {in_text} \n\n -> {out_text} \n\n => {final_output} \n\n' \
                                for (in_text, out_text, final_output) in zip(decoded_input_text, decoded_output_text, decoded_output_str)]
                writer.add_text('text_result/train', '\n\n'.join(decoded_pairs), train_step)
                writer.add_scalar('WER/train', np.mean(wer_list), train_step)
                logging_image = mel_tensor_to_plt_image(mel_tensor, decoded_input_text, train_step)
                writer.add_image('input_spectrogram/train', logging_image, train_step)
                print(f'Train Step {train_step}')
                # print(decoded_pairs)
                # writer.add_text('text_result', '', train_step)
                loss_list = list()
                wer_list = list()

            if train_step % CHECKPOINT_STEPS == 0:
                save_checkpoint(model, optimizer, train_step, writer.logdir, KEEP_LAST_ONLY)

            # break

        loss_test_list = list()
        wer_test_list = list()

        model.eval()
        for data in tqdm(dataset_test):
            mel_tensor, jamo_code_tensor, mel_lengths, jamo_lengths, mel_transformer_mask, speakers = data
            decoded_result = KOREAN_TABLE.decode_jamo_code_tensor(jamo_code_tensor, no_pad=True)
            speaker_code = speaker_table.speaker_name_to_code(speakers)
            _speakers = speaker_table.code_to_speaker_name(speaker_code)

            output_tensor = model((mel_tensor.to(cuda), 
                                mel_transformer_mask.to(cuda),
                                ))

            output_tensor = output_tensor.permute(1, 0, 2) # (N, S, E) => (T, N, C)

            loss = loss_criterion(output_tensor, 
                                jamo_code_tensor.to(cuda), 
                                (mel_lengths / 8).to(cuda), 
                                jamo_lengths.to(cuda))

            loss_test_list.append(loss.item())

            decoded_input_text = KOREAN_TABLE.decode_jamo_code_tensor(jamo_code_tensor)
            decoded_input_text = KOREAN_TABLE.decode_ctc_prediction(decoded_input_text)
            decoded_output_text = KOREAN_TABLE.decode_jamo_prediction_tensor(output_tensor)
            decoded_output_str = KOREAN_TABLE.decode_ctc_prediction(decoded_output_text)
            wer = KOREAN_TABLE.caculate_wer(decoded_input_text, decoded_output_str)
            wer_test_list.append(wer)

        decoded_pairs =  [f'** {in_text} \n\n -> {out_text} \n\n => {final_output} \n\n' \
                    for (in_text, out_text, final_output) in zip(decoded_input_text, decoded_output_text, decoded_output_str)]
        writer.add_scalar('ctc_loss/test', np.mean(loss_test_list), train_step)
        writer.add_scalar('WER/test', np.mean(wer_test_list), train_step)
        writer.add_text('text_result/test', '\n\n'.join(decoded_pairs), train_step)
        logging_image = mel_tensor_to_plt_image(mel_tensor, decoded_input_text, train_step)
        writer.add_image('input_spectrogram/test', logging_image, train_step)
                
