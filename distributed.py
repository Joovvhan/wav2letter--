
from models import TransformerSTT
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
import json

import torch

import numpy as np
import os

import sys

global_scope = sys.modules[__name__]

import argparse

from texts import KOREAN_TOKENS, KOREAN_TABLE

from mel2samp_waveglow import Mel2SampWaveglow

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'NanumGothic'

import torch.nn as nn

from prepare_batch import load_metadata, SpeakerTable, collate_function

from run import save_checkpoint, mel_tensor_to_plt_image, resume_training, load_checkpoint

import os

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.cuda.amp import autocast, GradScaler

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

TRAIN_METADATA_FILTERED_FILE = 'metadata_train_clean_filtered.csv'
TEST_METADATA_FILTERED_FILE = 'metadata_test_clean_filtered.csv'

LEN_TRAIN = 736153
LEN_TEST = 14890

# SR = 22050

KOREAN_PATTERN = re.compile('[^ㄱ-ㅎ|ㅏ-ㅣ|가-힣| .,!?]')

def process(rank, world_size, train_pairs, test_pairs, resume):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = rank

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_pairs,
                    num_replicas=world_size, rank=rank, shuffle=False)

    # dataset_train = DataLoader(train_pairs, batch_size=BATCH_SIZE, 
    #                            shuffle=True, num_workers=NUM_WORKERS,
    #                            collate_fn=collate_function,
    #                            pin_memory=True)

    dataset_train = DataLoader(train_pairs, batch_size=BATCH_SIZE, 
                               shuffle=False, num_workers=NUM_WORKERS,
                               collate_fn=collate_function,
                               pin_memory=True,
                               sampler=train_sampler)

    dataset_test = DataLoader(test_pairs, batch_size=BATCH_SIZE, 
                              shuffle=False, num_workers=NUM_WORKERS,
                              collate_fn=collate_function,
                              drop_last=True,
                              pin_memory=True)

    model = TransformerSTT(**model_parameters)
    # model = nn.DataParallel(model)
    model = model.to(device)
    model = DDP(model, find_unused_parameters=True, device_ids=[rank])
    # print(str(model))
    learning_rate = LR
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_criterion = nn.CTCLoss(zero_infinity=True)
    train_step = 0

    model, optimizer, train_step, writer = resume_training(resume, 
                                                           model, 
                                                           optimizer, 
                                                           train_step,
                                                           rank)

    scaler = GradScaler()

    loss_list = list()
    wer_list = list()

    for epoch in range(NUM_EPOCH):
        model.train()
        for data in tqdm(dataset_train):
            mel_tensor, jamo_code_tensor, mel_lengths, jamo_lengths, mel_transformer_mask, speakers = data

            # speaker_code = speaker_table.speaker_name_to_code(speakers)

            with autocast():
                output_tensor = model((mel_tensor.to(device), 
                                    mel_transformer_mask.to(device),
                                    ))

                output_tensor = output_tensor.permute(1, 0, 2) # (N, S, E) => (T, N, C)

                loss = loss_criterion(output_tensor, 
                                    jamo_code_tensor.to(device), 
                                    (mel_lengths // 8).to(device), 
                                    jamo_lengths.to(device))

            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_step += 1

            if rank == 0:

                decoded_input_text = KOREAN_TABLE.decode_jamo_code_tensor(jamo_code_tensor)
                decoded_input_text = KOREAN_TABLE.decode_ctc_prediction(decoded_input_text)
                decoded_output_text = KOREAN_TABLE.decode_jamo_prediction_tensor(output_tensor)
                decoded_output_str = KOREAN_TABLE.decode_ctc_prediction(decoded_output_text)

                wer = KOREAN_TABLE.caculate_wer(decoded_input_text, decoded_output_str)
                wer_list.append(wer)
                loss_list.append(loss.item())

                if len(loss_list) >= LOGGING_STEPS:
                    writer.add_scalar('ctc_loss/train', np.mean(loss_list), train_step)
                    decoded_pairs =  [f'** {in_text} \n\n -> {out_text} \n\n => {final_output} \n\n' \
                                    for (in_text, out_text, final_output) in zip(decoded_input_text, decoded_output_text, decoded_output_str)]
                    writer.add_text('text_result/train', '\n\n'.join(decoded_pairs), train_step)
                    writer.add_scalar('WER/train', np.mean(wer_list), train_step)
                    logging_image = mel_tensor_to_plt_image(mel_tensor, decoded_input_text, train_step)
                    writer.add_image('input_spectrogram/train', logging_image, train_step)
                    print(f'Train Step {train_step}')
                    loss_list = list()
                    wer_list = list()

                if train_step % CHECKPOINT_STEPS == 0:
                    save_checkpoint(model, optimizer, train_step, writer.logdir, KEEP_LAST_ONLY)
            

            # break

        if rank == 0:

            loss_test_list = list()
            wer_test_list = list()

            model.eval()
            for data in tqdm(dataset_test):
                mel_tensor, jamo_code_tensor, mel_lengths, jamo_lengths, mel_transformer_mask, speakers = data

                with autocast():
                    output_tensor = model((mel_tensor.to(device), 
                                        mel_transformer_mask.to(device),
                                        ))

                    output_tensor = output_tensor.permute(1, 0, 2) # (N, S, E) => (T, N, C)

                    loss = loss_criterion(output_tensor, 
                                        jamo_code_tensor.to(device), 
                                        (mel_lengths // 8).to(device), 
                                        jamo_lengths.to(device))

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-resume', metavar='-r', type=str,
                        help='resume train', default=None)

    args = parser.parse_args()

    train_pairs = load_metadata(TRAIN_METADATA_FILTERED_FILE)
    test_pairs = load_metadata(TEST_METADATA_FILTERED_FILE)

    train_pairs.sort(key=lambda x: x[6])
    test_pairs.sort(key=lambda x: x[6])

    # (wav_file, clean_script, clean_jamos, tag, len(clean_script), len(clean_jamos), wav_file_dur)
    speaker_table = SpeakerTable(set([pair[3] for pair in train_pairs] + [pair[3] for pair in test_pairs]))

    print(len(speaker_table))

    def is_valid(string):
        if string in ['prosem_f', 'prosem_m', 'kss']:
            return True
        elif 'acriil' in string or 'clova' in string:
            return True
        else:
            # return False
            return True

    print(len(train_pairs), len(test_pairs))

    train_pairs = list(filter(lambda x: is_valid(x[3]), train_pairs))
    test_pairs = list(filter(lambda x: is_valid(x[3]), test_pairs))

    print('>>>', len(train_pairs), len(test_pairs))

    print(KOREAN_TOKENS)

    world_size = 2
    mp.spawn(process,
        args=(world_size, train_pairs, test_pairs, args.resume),
        nprocs=world_size,
        join=True)

    dist.destroy_process_group()
