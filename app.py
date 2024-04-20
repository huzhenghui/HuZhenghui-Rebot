import gradio as gr
import os
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

base_path = './HuZhenghui-Robot/'

# Clone the repository
clone_command = f'git clone https://code.openxlab.org.cn/HuZhenghui/HuZhenghui-Robot.git {base_path}'
clone_process = subprocess.Popen(clone_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
out, err = clone_process.communicate()
print(out.decode('utf-8'))
print(err.decode('utf-8'))

# Pull from the repository
pull_command = f'cd {base_path} && git lfs pull'
pull_process = subprocess.Popen(pull_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
out, err = pull_process.communicate()
print(out.decode('utf-8'))
print(err.decode('utf-8'))

tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()
model = model.eval()

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="HuZhenghui-Robot",
                description="""
我是胡争辉的个人小助手.  
                 """,
                 ).queue(1).launch()
