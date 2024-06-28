// Copyright (c) 2023 ZhaoDe Wang
//               2024 Chengdong Liang(liangchengdongd@qq.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mnn_llm_tokenizer/tokenizer.h"

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " tokenizer.txt" << std::endl;
        return 0;
    }
    std::string tokenizer_path = argv[1];
    std::unique_ptr<Tokenizer> tokenizer(
        Tokenizer::createTokenizer(tokenizer_path));
    const std::string system_str = "你好";
    // const std::string user_str = "<|endoftext|>";
    // const std::string query = "\n<|im_start|>system\n" + system_str + "<|im_end|>\n<|im_start|>\n" + user_str + "<|im_end|>\n<|im_start|>assistant\n";
    const std::string query = system_str + "\n";
    printf("query = %s\n", query.c_str());
    auto tokens = tokenizer->encode(query);

    std::string decode_str = "";
    printf("encode tokens = [ ");
    for (auto token : tokens) {
        printf("%d ", token);
        decode_str += tokenizer->decode(token);
    }
    printf("]\n");
    printf("decode str = %s\n", decode_str.c_str());
    return 0;
}
