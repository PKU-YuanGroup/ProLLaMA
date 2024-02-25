# ProLLaMA
A Protein Large Language Model for Multi-Task Protein Language Processing

Large Language Models (LLMs), including GPT-x and LLaMA2, have achieved remarkable performance in multiple Natural Language Processing (NLP) tasks. 
Under the premise that protein sequences constitute the protein language, Protein Large Language Models (ProLLMs) trained on protein corpora excel at de novo protein sequence generation.
However, as of now, unlike LLMs in NLP, no ProLLM is capable of multiple tasks in the **Protein Language Processing (PLP)** field.
We introduce **a training framework** to transform any general LLM into a ProLLM capable of **handling multiple PLP tasks**. Specifically, our framework utilizes low-rank adaptation and employs a two-stage training approach, and it is distinguished by its universality, low overhead, and scalability. Through training under this framework, we propose **the ProLLaMA model**, the first known ProLLM to handle multiple PLP tasks simultaneously.
Experiments show that ProLLaMA achieves state-of-the-art results in the unconditional protein sequence generation task. In the controllable protein sequence generation task, ProLLaMA can **design novel proteins with desired functionalities**. In the protein property prediction task, ProLLaMA achieves nearly 100\% accuracy across many categories. The latter two tasks are beyond the reach of other ProLLMs

## 1.Install Requirements

```bash
git clone https://github.com/Lyu6PosHao/ProLLaMA.git
cd ProLLaMA
pip install -r requirements.txt
```

## 2.Download Model
Download from [HuggingFace](https://huggingface.co/GreatCaptainNemo/ProLLaMA)

## 3.Usage
As ProLLaMA's architecture is the same as LLaMA2, you can use ProLLaMA like using LLaMA2. 

Here are three ways.
### Commandline

```bash
python main.py --model "GreatCaptainNemo/ProLLaMA" --interactive
#You can replace the model_path with your local path
#Use "python main.py -h" for more details
```

### Python
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
##You can replace the model_path with your local path
tokenizer = AutoTokenizer.from_pretrained("GreatCaptainNemo/ProLLaMA", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("GreatCaptainNemo/ProLLaMA", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("GreatCaptainNemo/ProLLaMA")
print("####Enter 'exit' to exit.")
while True:
    messages = []
    user=str(input("Input:"))
    if user.strip()=="exit":
        break
    messages.append({"role": "user", "content": user})
    response = model.chat(tokenizer, messages)
    print("Output:", response)
```

### LLaMA-Factory
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
python ./src/cli_demo.py \
      --model_name_or_path /path_to_your_model \
      --template llama2
```

## 4.Demo

### Input Format
```text
[Generate by superfamily] Superfamily=<xxx>
or
[Determine superfamily] Seq=<yyy>
```

### Cases
TBD



