# llm_tokenizer

编译:
```bash
mkdir build
cd build
cmake .. && make
```

导出tokenizer文件：
```bash
python tools/export_token.py \
    --path /path/to/Qwen1.5-1.8B-Chat \
    --token_file ./resource/qwen2-1.8b_token.txt \
    --model_name Qwen1_5-1_8B-Chat
```

测试:
```bash
./build/bin/tokenizer_main resource/qwen2-1.8b_token.txt
```
