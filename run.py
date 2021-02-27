
from models import TransformerSTT

import csv 
from torch.utils.data import DataLoader
from tqdm import tqdm

import random
import re
import json
import math
import shutil

import torch

import numpy as np
import os

import sys

global_scope = sys.modules[__name__]

import argparse
from glob import glob 
from collections import OrderedDict

from texts import KOREAN_TOKENS, KOREAN_TABLE

from mel2samp_waveglow import Mel2SampWaveglow

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'NanumGothic'

import jamotools

import torch.nn as nn

from tensorboardX import SummaryWriter

from prepare_batch import load_metadata, SpeakerTable, collate_function

CONFIGURATION_FILE='config.json'

with open(CONFIGURATION_FILE) as f:
    data = f.read()
    json_info = json.loads(data)

    mel_config = json_info["mel_config"]
    MEL2SAMPWAVEGLOW = Mel2SampWaveglow(**mel_config)

    hp = json_info["hp"]

    for key in hp:
        setattr(global_scope, key, hp[key])
        # print(f'{key} == {hp[key]}')

    model_parameters = json_info["mp"]

# TRAIN_METADATA_FILE = 'metadata_train_clean.csv'
# TEST_METADATA_FILE = 'metadata_test_clean.csv'

TRAIN_METADATA_FILTERED_FILE = 'metadata_train_clean_filtered.csv'
TEST_METADATA_FILTERED_FILE = 'metadata_test_clean_filtered.csv'

LEN_TRAIN = 736153
LEN_TEST = 14890

# SR = 22050

KOREAN_PATTERN = re.compile('[^ㄱ-ㅎ|ㅏ-ㅣ|가-힣| .,!?]')

def load_checkpoint(model, optimizer, path, map_location=None):

    checkpoint_files = sorted(glob(os.path.join(path, '*.pt')))
    assert len(checkpoint_files) > 0, f'No checkpoint inside {path}'

    checkpoint_file = checkpoint_files[-1]

    if map_location is not None:
        # https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

        if map_location == 'cpu':
            new_model_checkpoint = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                key = k.replace('module.', '')
                new_model_checkpoint[key] = v
            checkpoint['model_state_dict'] = new_model_checkpoint

    else:
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

def resume_training(resume, model, optimizer, step, rank=0):
    if resume is not None:
        if resume == 'latest':
            logging_folders = sorted(glob('runs/*'))
            assert len(logging_folders) > 0, f'No folder exists inside ./runs'
            logging_path = logging_folders[-1]
        else:
            logging_path = os.path.join('runs', resume)
    else:
        logging_path = ''

    if os.path.isdir(logging_path):
        summary_writer = SummaryWriter(logging_path)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        model, optimizer, step = load_checkpoint(model, optimizer, summary_writer.logdir, map_location)
    
    else:
        if rank == 0:
            print(f'Logging path {logging_path} does not exist')
            os.mkdir(logging_path)
            summary_writer = SummaryWriter(logging_path)
            shutil.copy('config.json', summary_writer.logdir)
        else:
            summary_writer = None

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

    train_pairs.sort()
    test_pairs.sort()

    print('>>>', len(train_pairs), len(test_pairs))

    # dataset_train = DataLoader(train_pairs, batch_size=BATCH_SIZE, 
    #                            shuffle=True, num_workers=NUM_WORKERS,
    #                            collate_fn=collate_function)

    dataset_train = DataLoader(train_pairs, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=NUM_WORKERS,
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
                
