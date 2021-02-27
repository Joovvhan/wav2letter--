from collections import Iterable
import Levenshtein 
import torch
import numpy as np

# https://pypi.org/project/jamotools/
# Hangul Jamo: Consist of Choseong, Jungseong, Jongseong. 
# It is divided mordern Hangul and old Hangul that does not use in nowadays. 
# Jamotools supports modern Hangul Jamo area.
# 1100 ~ 1112 (Choseong)
# 1161 ~ 1175 (Jungseong)
# 11A8 ~ 11C2 (Jongseong)

PAD = '_'

CHOSEONGS = [chr(i) for i in range(0x1100, 0x1112 + 1)]
JUNGSEONGS = [chr(i) for i in range(0x1161, 0x1175 + 1)]
JONGSEONGS = [chr(i) for i in range(0x11A8, 0x11C2 + 1)]
PUNCTUATIONS = ['.', '!', '?', ',',  ' ']
SPECIALS = ['<s>', '</s>']

KOREAN_TOKENS = list()
KOREAN_TOKENS.append(PAD)
KOREAN_TOKENS.extend(PUNCTUATIONS)
KOREAN_TOKENS.extend(CHOSEONGS)
KOREAN_TOKENS.extend(JUNGSEONGS)
KOREAN_TOKENS.extend(JONGSEONGS)
KOREAN_TOKENS.extend(SPECIALS)

print(KOREAN_TOKENS)

class KoreanTable():

    def __init__(self, tokens):
        self.tokens = tokens
        self.korean_dict = {c: i for i, c in enumerate(self.tokens)}

    def __get__(self, input_code):

        if isinstance(input_code, str):
            return self.korean_dict[input_code]
        elif isinstance(input_code, int):
            return self.korean_dict[input_code]
        else:
            assert False, f'Invalid input ({input_code}) for KoreanTable '

    def jamo_to_jamo_code(self, jamo, append_specials=False):
        
        jamo_code = [self.korean_dict[c] for c in jamo]
        
        if append_specials:
            jamo_code = [self.korean_dict['<s>'], *jamo_code, self.korean_dict['</s>']]

        return jamo_code

    def jamo_code_to_jamo(self, jamo_code):
        return [self.tokens[i] for i in jamo_code]
    
    def decode_jamo_code_tensor(self, jamo_code_tensor, no_pad=False):
        B = jamo_code_tensor.shape[0]

        decoded_result = list()

        for i in range(B):
            decoded_string = ''.join(self.jamo_code_to_jamo(jamo_code_tensor[i, :]))
            if no_pad:
                decoded_string = decoded_string.replace(PAD, '')
            decoded_result.append(decoded_string)
        
        return decoded_result

    def decode_jamo_prediction_tensor(self, jamo_p_tensor, no_pad=False):
        B = jamo_p_tensor.shape[1] # (T, N, C)
        jamo_p_tensor = jamo_p_tensor.permute(1, 0, 2) # (T, N, C) =>  (N, T, C)
        jamo_code_tensor = torch.argmax(jamo_p_tensor, dim=-1)

        decoded_result = list()

        for i in range(B):
            decoded_string = ''.join(self.jamo_code_to_jamo(jamo_code_tensor[i, :]))
            if no_pad:
                decoded_string = decoded_string.replace(PAD, '')
            decoded_result.append(decoded_string)
        
        return decoded_result
    
    def decode_ctc_prediction(self, ctc_input_list):

        decoded_result = list()

        for ctc_input in ctc_input_list:
            decoded = ''
            for c in ctc_input:
                if c != '_':
                    if len(decoded) == 0:
                        decoded += c 
                    elif c != decoded[-1]:
                        decoded += c 
                    
            decoded_result.append(decoded)

        return decoded_result

    def caculate_wer(self, input_text_list, output_text_list):

        wer_list = list()
        
        for i, o in zip(input_text_list, output_text_list):
            i = i.replace(' ', '').replace('<s>', '').replace('</s>', '')
            # o = o.replace(' ', '').replace('<s>', '').replace('</s>', '')
            o = o.replace(' ', '').replace('<s>', '').split('</s>')[0]
            wer = Levenshtein.distance(i, o) / len(i)
            wer_list.append(wer)
        
        wer_mean = np.mean(wer_list)

        return wer_mean

KOREAN_TABLE = KoreanTable(KOREAN_TOKENS)