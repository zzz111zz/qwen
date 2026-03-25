# Qwen1.5-0.5B-Chat 本地部署与 WebUI 搭建指南 🚀

本文将详细介绍如何在本地环境下载 Qwen1.5-0.5B-Chat 模型，并基于 Gradio 搭建一个可交互的聊天 WebUI。

## 一、环境准备

### 1. 安装依赖库

首先需要安装项目所需的 Python 库，打开终端执行以下命令:

```
pip install torch transformers gradio modelscope
```

`torch`: PyTorch 深度学习框架，用于模型推理

`transformers`: HuggingFace 库，用于加载和运行大语言模型

`gradio`: 用于快速构建 Web 交互界面

`modelscope`: 阿里云模型库，用于下载模型文件

## 二、下载 Qwen1.5-0.5B-Chat 模型

### 1. 模型下载代码

创建 `1.iypnb` 文件，写入以下代码

```
from modelscope import snapshot_download

# 下载模型到指定目录
model_dir = snapshot_download(
    'qwen/Qwen1.5-0.5B-chat',
    cache_dir='D:/Large Model/Qwen1.5'  # 可修改为你的本地存储路径
)

print("模型下载完成，路径：", model_dir)
```

### 2. 代码说明

`snapshot_download`: 从 ModelScope 下载完整模型文件

`cache_dir`: 模型保存的本地路径，建议使用**绝对路径**（如 `D:/Large Model/Qwen1.5`）

执行后，模型将被下载到 `D:/Large Model/Qwen1.5/qwen/Qwen1___5-0___5B-chat` 目录下

## 三、模型推理测试

### 1. 测试代码

创建 `web_demo.py` 文件，验证模型是否能正常加载和推理：

```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 自动选择设备（GPU 优先）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = r"D:\Large Model\Qwen1.5\qwen\Qwen1___5-0___5B-chat"  # 与下载路径一致

# 加载模型与分词器
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 构造对话 Prompt
prompt = "简单介绍一下你自己"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# 应用对话模板
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# 生成回复
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 解码并输出结果
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

### 2. 代码说明

`device_map="auto"`: 自动分配模型到可用的 GPU/CPU

`apply_chat_template`: 按照 Qwen 模型要求格式化对话输入

`max_new_tokens=512`: 限制生成回复的最大长度

运行后将输出模型对 Prompt 的回复，验证模型功能正常

## 四、搭建 Gradio WebUI 聊天界面

### 1. WebUI 完整代码

```
from argparse import ArgumentParser
from threading import Thread
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# 默认模型路径（与下载路径一致）
DEFAULT_CKPT_PATH = "D:\Large Model\Qwen1.5\qwen\Qwen1___5-0___5B-chat"

def _get_args():
    parser = ArgumentParser(description="Qwen1.5-Instruct web chat demo.")
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH, help="模型路径")
    parser.add_argument("--cpu-only", action="store_true", help="仅使用 CPU 运行")
    parser.add_argument("--share", action="store_true", default=False, help="生成公开链接")
    parser.add_argument("--inbrowser", action="store_true", default=False, help="自动打开浏览器")
    parser.add_argument("--server-port", type=int, default=8000, help="服务端口")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="服务地址")
    return parser.parse_args()

def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, resume_download=True)
    device_map = "cpu" if args.cpu_only else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path, torch_dtype="auto", device_map=device_map, resume_download=True
    ).eval()
    model.generation_config.max_new_tokens = 2048
    return model, tokenizer

def _chat_stream(model, tokenizer, query, history):
    conversation = []
    for q, r in history:
        conversation.append({"role": "user", "content": q})
        conversation.append({"role": "assistant", "content": r})
    conversation.append({"role": "user", "content": query})
    input_text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
    thread = Thread(target=model.generate, kwargs={**inputs, "streamer": streamer})
    thread.start()
    for new_text in streamer:
        yield new_text

def _launch_demo(args, model, tokenizer):
    def predict(_query, _chatbot, _task_history):
        _chatbot.append((_query, ""))
        full_response = ""
        for new_text in _chat_stream(model, tokenizer, _query, _task_history):
            full_response += new_text
            _chatbot[-1] = (_query, full_response)
            yield _chatbot
        _task_history.append((_query, full_response))

    def regenerate(_chatbot, _task_history):
        if not _task_history:
            yield _chatbot
            return
        item = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot, _task_history)

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        return _chatbot

    with gr.Blocks() as demo:
        gr.Markdown("""<p align="center"><img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/assets/logo/qwen1.5_logo.png" style="height: 120px"/></p>""")
        gr.Markdown("""<center><font size=3>This WebUI is based on Qwen1.5-Instruct, developed by Alibaba Cloud.</center>""")
        chatbot = gr.Chatbot(label="Qwen")
        query = gr.Textbox(lines=2, label="Input")
        task_history = gr.State([])
        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")
        submit_btn.click(predict, [query, chatbot, task_history], [chatbot])
        submit_btn.click(lambda: "", [], [query])
        empty_btn.click(reset_state, [chatbot, task_history], [chatbot])
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot])
        gr.Markdown("""<font size=2>Note: This demo is governed by the original license of Qwen1.5.</font>""")
    demo.queue().launch(
        share=args.share, inbrowser=args.inbrowser,
        server_port=args.server_port, server_name=args.server_name
    )

def main():
    args = _get_args()
    model, tokenizer = _load_model_tokenizer(args)
    _launch_demo(args, model, tokenizer)

if __name__ == "__main__":
    main()
```

### 2. 启动 WebUI

在终端执行以下命令启动服务：

```
# 基础启动（本地访问，端口 8000）
python web_demo.py
```

### 3. 界面功能说明

**聊天窗口**: 显示用户与模型的对话历史

**输入框**: 输入提问内容

**Submit (发送)**: 提交问题，模型流式输出回复

**Regenerate (重试)**: 重新生成上一条回复

**Clear History (清除历史)**: 清空所有对话记录

启动后访问 `http://127.0.0.1:8000` 即可使用聊天界面

## 五、常见问题与注意事项

1. **模型路径错误**: 确保 `DEFAULT_CKPT_PATH` 与实际下载路径完全一致，Windows 下建议使用 `r"路径"` 避免转义字符问题
2. **显存不足**: 0.5B 模型占用显存较小，若仍提示 OOM，可添加 `--cpu-only` 参数切换到 CPU 推理
3. **端口占用**: 修改 `--server-port` 参数为其他未被占用的端口（如 8080、8081）
4. **流式输出**: 代码中使用 `TextIteratorStreamer` 实现了打字机效果的流式回复

## 六、项目结构参考

```
Qwen1.5/
├── 1.ipynb    # 模型下载脚本
├── web_demo.py        # Gradio WebUI 脚本
└── qwen/
    └── Qwen1___5-0___5B-chat/  # 模型文件目录
```

## 七、效果展示![qwen1.5](D:\Large Model\Qwen1.5\qwen1.5.JPG)