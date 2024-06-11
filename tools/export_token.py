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


def has_model_file(directory, suffix='.model'):
    for filename in os.listdir(directory):
        if filename.endswith(suffix):
            return directory + '/' + filename
    return None


class ExportToken:
    def __init__(self, model_path, token_file, model_name, tokenzier_class=None) -> None:
        super().__init__()
        # get tokenizer_class for huggingface models
        with open(os.path.join(model_path, "tokenizer_config.json"), 'r') as f:
            tokenizer_config = json.load(f)
        tokenizer_class = tokenizer_config.get('tokenizer_class')
        if tokenizer_class in ["PreTrainedTokenizerFast", "PreTrainedTokenizer"]:
            is_hf_tokenizer = True
        else:
            is_hf_tokenizer = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.sp_model = None
        if not is_hf_tokenizer:
            sp_model = has_model_file(model_path, '.model')
            if os.path.exists(sp_model):
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

    def export_tokenizer(self):
        file_path = os.path.join(self.token_file, "tokenizer.txt")
        if self.sp_model is not None:
            # senetencepiece
            print('# senetencepiece tokenier')
            NORMAL = 1; UNKNOWN = 2; CONTROL = 3
            USER_DEFINED = 4; UNUSED = 5; BYTE = 6
            fp = open(file_path, "w", encoding="utf8")
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
                fp.write(f'{token_encode} {score} {type}\n')
            fp.close()
        elif hasattr(self.tokenizer, 'mergeable_ranks'):
            print('# tiktoken tokenier')
            # tikton
            with open(file_path, "w", encoding="utf8") as fp:
                for k, v in self.tokenizer.mergeable_ranks.items():
                    line = base64.b64encode(k).decode("utf8") + "\n"
                    fp.write(line)
                if hasattr(self.tokenizer, 'special_tokens'):
                    for k, v in self.tokenizer.special_tokens.items():
                        line = base64.b64encode(k.encode("utf-8")).decode("utf8") + "\n"
                        fp.write(line)
        elif self.merge_txt is not None:
            # huggingface tokenizer
            print("# huggingface tokenizer with merges.txt")
            merge_list = []
            vocab = self.tokenizer.get_vocab()
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
                fp.write(f'{len(vocab_list)} {len(merge_list)}\n')
                for v in vocab_list:
                    fp.write(v + '\n')
                for m in merge_list:
                    fp.write(m)
        else:
            # huggingface tokenizer
            print("# huggingface tokenizer")
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
            with open(file_path, "w", encoding="utf8") as fp:
                vocab = self.tokenizer.get_vocab()
                vocab_list = ['<unk>' for i in range(len(vocab))]
                for k, v in vocab.items():
                    try:
                        vocab_list[int(v)] = bytes([unicode_to_byte(ord(c)) for c in k]).decode('utf-8', errors='ignore')
                    except:
                        vocab_list[int(v)] = k
                for v in vocab_list:
                    line = base64.b64encode(v.encode('utf-8')).decode("utf8") + "\n"
                    fp.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='THUDM/chatglm-6b', required=True,
                        help='path(`str` or `os.PathLike`):\nCan be either:'
                        '\n\t- A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]'
                        '\n\t- A path to a *directory* clone from repo like `../chatglm-6b`.')
    parser.add_argument('--token_file', type=str, default='.', required=False,
                        help='path to save token file')
    parser.add_argument('--model_name', type=str, default='llama2-7B-chat', required=False)
    args = parser.parse_args()
    export_token = ExportToken(args.path, args.token_file, args.model_name)
    export_token.export_tokenizer()
