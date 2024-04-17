import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from openxlab.model import download

base_path = './HuZhenghui-Robot/'
os.system(f'git clone https://code.openxlab.org.cn/HuZhenghui/HuZhenghui-Robot.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True)

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="HuZhenghui-Robot",
                description="""
我是胡争辉的个人小助手.  
                 """,
                 ).queue(1).launch()
