# Copyright (c) 2023 ZhaoDe Wang
#               2024 Chengdong Liang(liangchengdongd@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import os
import json
import argparse
import sentencepiece as spm

from transformers import AutoTokenizer


MODEL_CLASSES = {
        'Qwen-7B-Chat': "qwen",
        'Qwen-1_8B-Chat': "qwen",
        'Qwen-1_8B': "qwen",
        'Qwen-VL-Chat': "qwen",
        'Qwen1_5-0_5B-Chat': "qwen2",
        'Qwen1_5-1_8B-Chat': "qwen2",
        'Qwen1_5-4B-Chat': "qwen2",
        'Qwen1_5-7B-Chat': "qwen2",
        'Qwen2-0_5B-Instruct': "qwen2",
        'Qwen2-1_5B-Instruct': "qwen2",
        'Qwen2-7B-Instruct': "qwen2",
        'Llama-2-7b-chat': "llama2",
        'Llama-3-8B-Instruct': "llama3",
        'internlm-chat-7b': "llama2",
        'deepseek-llm-7b-chat': "llama2",
    }


def has_model_file(directory, suffix='.model'):
    for filename in os.listdir(directory):
        if filename.endswith(suffix):
            return directory + '/' + filename
    return None


class ExportToken:
    def __init__(self, model_path, token_file, model_name) -> None:
        super().__init__()
        self.stop_ids = []
        self.MAGIC_NUMBER = 430
        # TOKENIZER TYPE
        self.tokenizer_type = {"sentencepiece": 0, "tiktoken": 1, "bert": 2, "huggingface": 3}
        # get tokenizer_class for huggingface models
        with open(os.path.join(model_path, "tokenizer_config.json"), 'r') as f:
            tokenizer_config = json.load(f)
        tokenizer_class = tokenizer_config.get('tokenizer_class')
        if tokenizer_class in ["PreTrainedTokenizerFast", "PreTrainedTokenizer"]:
            is_hf_tokenizer = True
        else:
            is_hf_tokenizer = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.encode("你好\n")
        self.sp_model = None
        if not is_hf_tokenizer:
            sp_model = has_model_file(model_path, '.model')
            if sp_model is not None and os.path.exists(sp_model):
                try:
                    self.sp_model = spm.SentencePieceProcessor(model_file=sp_model)
                    print("sentencepiece model is found")
                except:
                    print("sentencepiece model is not found")

        merge_txt = os.path.join(model_path, "merges.txt")
        if os.path.exists(merge_txt):
            self.merge_txt = merge_txt
        else:
            self.merge_txt = None
        self.token_file = token_file
        self.model_name = model_name

        if "llama2" == MODEL_CLASSES[model_name]:
            self.stop_ids.append(self.tokenizer.eos_token_id)
            if os.path.exists(os.path.join(model_path, "generation_config.json")):
                with open(os.path.join(model_path, "generation_config.json"), 'r') as f:
                    generation_config = json.load(f)
                self.stop_ids.append(self.model.generate_config['eos_token_id'])
        elif "llama3" == MODEL_CLASSES[model_name]:
            self.stop_ids.append(self.tokenizer.eos_token_id)
            self.stop_ids.append(self.tokenizer.convert_tokens_to_ids('<|eot_id|>'))
        elif "qwen2"  == MODEL_CLASSES[model_name]:
            self.stop_ids.append(self.tokenizer.eos_token_id)
            if os.path.exists(os.path.join(model_path, "generation_config.json")):
                with open(os.path.join(model_path, "generation_config.json"), 'r') as f:
                    generation_config = json.load(f)
                for id in generation_config['eos_token_id']:
                    self.stop_ids.append(id)
        elif "qwen" == MODEL_CLASSES[model_name]:
            self.stop_ids.append(self.tokenizer.im_end_id)


    def write_line(self, fp, *args):
        for arg in args:
            for token in arg:
                fp.write(str(token) + ' ')
        fp.write('\n')

    def write_header(self, fp, type, speicals, prefix = []):
        fp.write(f'{self.MAGIC_NUMBER} {type}\n')
        fp.write(f'{len(speicals)} {len(self.stop_ids)} {len(prefix)}\n')
        self.write_line(fp, speicals, self.stop_ids, prefix)

    def export_tokenizer(self):
        file_path = os.path.join(self.token_file)
        special_list = []
        if hasattr(self.tokenizer, 'added_tokens_decoder'):
            special_list = list(self.tokenizer.added_tokens_decoder.keys())
        if hasattr(self.tokenizer, 'special_tokens'):
            for k, v in self.tokenizer.special_tokens.items():
                special_list.append(v)
        if hasattr(self.tokenizer, 'gmask_token_id'):
            special_list.append(self.tokenizer.gmask_token_id)
        vocab_list = []
        prefix_list = []
        if hasattr(self.tokenizer, 'get_prefix_tokens'):
            prefix_list = self.tokenizer.get_prefix_tokens()

        if self.sp_model is not None:
            # senetencepiece
            print('# senetencepiece tokenier')
            NORMAL = 1; UNKNOWN = 2; CONTROL = 3
            USER_DEFINED = 4; UNUSED = 5; BYTE = 6
            for i in range(self.sp_model.GetPieceSize()):
                token = self.sp_model.IdToPiece(i)
                score = self.sp_model.GetScore(i)
                type = NORMAL
                if self.sp_model.IsUnknown(i):
                    type = UNKNOWN
                elif self.sp_model.IsControl(i):
                    type = CONTROL
                elif self.sp_model.IsUnused(i):
                    type = UNUSED
                elif self.sp_model.IsByte(i):
                    type = BYTE
                if self.model_name == 'Chatglm_6b':
                    if '<n>' in token: token = '\n'
                    if '<|tab|>' in token: token = '\t'
                    if '<|blank_' in token: token = ' ' * int(token[8:token.find('|>')])
                if '▁' in token: token = token.replace('▁', ' ')
                token_encode = base64.b64encode(token.encode("utf-8")).decode("utf8")
                vocab_list.append(f'{token_encode} {score} {type}\n')
            with open(file_path, "w", encoding="utf8") as fp:
                self.write_header(fp, self.tokenizer_type['sentencepiece'], special_list, prefix_list)
                fp.write(f'{len(vocab_list)}\n')
                for vocab in vocab_list:
                    fp.write(vocab)
        elif hasattr(self.tokenizer, 'mergeable_ranks'):
            print('# tiktoken tokenier')
            # tikton
            vocab_list = []
            for k, v in self.tokenizer.mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + "\n"
                vocab_list.append(line)
            if hasattr(self.tokenizer, 'special_tokens'):
                for k, v in self.tokenizer.special_tokens.items():
                    line = base64.b64encode(k.encode("utf-8")).decode("utf8") + "\n"
                    vocab_list.append(line)
            if hasattr(self.tokenizer, 'added_tokens_decoder'):
                for k, v in self.tokenizer.added_tokens_decoder.items():
                    line = base64.b64encode(v.__str__().encode("utf-8")).decode("utf8") + "\n"
                    vocab_list.append(line)
            with open(file_path, "w", encoding="utf8") as fp:
                self.write_header(fp, self.tokenizer_type['tiktoken'], special_list, prefix_list)
                fp.write(f'{len(vocab_list)}\n')
                for vocab in vocab_list:
                    fp.write(vocab)
        elif self.merge_txt is not None:
            # huggingface tokenizer
            merge_list = []
            vocab = self.tokenizer.get_vocab()
            if hasattr(self.tokenizer, 'added_tokens_decoder'):
                special_list = list(self.tokenizer.added_tokens_decoder.keys())
            vocab_list = ['<unk>' for i in range(len(vocab))]
            # load vocab
            for k, v in vocab.items():
                vocab_list[int(v)] = k
            # load merge
            with open(self.merge_txt, 'rt') as merge:
                for line in merge.readlines():
                    merge_list.append(line)
            # write to tokenizer.txt
            with open(file_path, "w", encoding="utf8") as fp:
                self.write_header(fp, self.tokenizer_type['huggingface'], special_list)
                fp.write(f'{len(vocab_list)} {len(merge_list)}\n')
                for v in vocab_list:
                    fp.write(v + '\n')
                for m in merge_list:
                    fp.write(m)
        else:
            print('# other tiktoken tokenier')
            # other tikton
            def unicode_to_byte(u: int):
                if u >= 256 and u <= 288:
                    return u - 256
                if u >= 289 and u <= 322:
                    return u - 162
                if u == 323:
                    return 173
                if u == 65372: # |
                    return 124
                if u == 9601:  # _
                    return 95
                return u
            vocab = self.tokenizer.get_vocab()
            vocab_list = ['<unk>' for i in range(len(vocab))]
            for k, v in vocab.items():
                try:
                    vocab_list[int(v)] = bytes([unicode_to_byte(ord(c)) for c in k]).decode('utf-8', errors='ignore')
                except:
                    vocab_list[int(v)] = k
            if hasattr(self.tokenizer, 'added_tokens_decoder'):
                special_list = list(self.tokenizer.added_tokens_decoder.keys())
            with open(file_path, "w", encoding="utf8") as fp:
                self.write_header(fp, self.tokenizer_type['tiktoken'], special_list)
                fp.write(f'{len(vocab_list)}\n')
                for v in vocab_list:
                    line = base64.b64encode(v.encode('utf-8')).decode("utf8") + "\n"
                    fp.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='THUDM/chatglm-6b', required=True,
                        help='path(`str` or `os.PathLike`):\nCan be either:'
                        '\n\t- A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]'
                        '\n\t- A path to a *directory* clone from repo like `../chatglm-6b`.')
    parser.add_argument('--token_file', type=str, default='./token.txt', required=False,
                        help='path to save token file')
    parser.add_argument('--model_name', type=str, choices=MODEL_CLASSES.keys(), required=True)
    args = parser.parse_args()
    export_token = ExportToken(args.path, args.token_file, args.model_name)
    export_token.export_tokenizer()
