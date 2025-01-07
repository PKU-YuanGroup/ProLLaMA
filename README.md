
<h2 align="center"> <a href="hhttps://arxiv.org/abs/2402.16445">ProLLaMA: A Protein Language Model for Multi-Task Protein Language Processing</a></h2>
<h5 align="center">
    
[![arXiv](https://img.shields.io/badge/Arxiv-2402.16445-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2402.16445)
[![Model1](https://img.shields.io/badge/🤗-Model1_Download-blue.svg)](https://huggingface.co/GreatCaptainNemo/ProLLaMA_Stage_1)
[![Model2](https://img.shields.io/badge/🤗-Model2_Download-blue.svg)](https://huggingface.co/GreatCaptainNemo/ProLLaMA)
[![Dataset](https://img.shields.io/badge/🤗-Dataset_Download-blue.svg)](https://huggingface.co/datasets/GreatCaptainNemo/instruction_dataset)
<!-- [![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Lyu6PosHao/ProLLaMA/blob/main/LICENSE) --> <br>

</h5>

## 📣 News
* [2025/01/07] Update some training codes for easier usage. Details in [Logs](https://github.com/PKU-YuanGroup/ProLLaMA?tab=readme-ov-file#logs).
* [2025/01/01] We propose [HME](https://github.com/Lyu6PosHao/HME), a multimodal multitask Chemical LLMs.
* [2024/07/17] Update a new version of the paper.
* [2024/06/27] Release the codes for pretraining (Stage1) and instruction_tuning (Stage2). See [Quick Train](https://github.com/PKU-YuanGroup/ProLLaMA?tab=readme-ov-file#%EF%B8%8Fquick-train).
* [2024/06/08] Opensource the instruction dataset on [HuggingFace](https://huggingface.co/datasets/GreatCaptainNemo/instruction_dataset)
* [2024/04/25] Upload ProLLaMA_Stage_1 to [HuggingFace](https://huggingface.co/GreatCaptainNemo/ProLLaMA_Stage_1). More information is in [Others](https://github.com/PKU-YuanGroup/ProLLaMA?tab=readme-ov-file#%EF%B8%8Fothers).
* [2024/04/10] Add a script (in /scripts/mutation.py) to meature mutation effects.
* [2024/02.29] Update the /scripts/infer.py to fix bugs.




## 🗝️ Abstract
Large Language Models (LLMs), including GPT-x and LLaMA2, have achieved remarkable performance in multiple Natural Language Processing (NLP) tasks. 
Under the premise that protein sequences constitute the protein language, Protein Large Language Models (ProLLMs) trained on protein corpora excel at de novo protein sequence generation.
However, as of now, unlike LLMs in NLP, no ProLLM is capable of multiple tasks in the **Protein Language Processing (PLP)** field.
We introduce **a training framework** to transform any general LLM into a ProLLM capable of **handling multiple PLP tasks**. Specifically, our framework utilizes low-rank adaptation and employs a two-stage training approach, and it is distinguished by its universality, low overhead, and scalability. Through training under this framework, we propose **the ProLLaMA model**, the first known ProLLM to handle multiple PLP tasks simultaneously.
Experiments show that ProLLaMA achieves state-of-the-art results in the unconditional protein sequence generation task. In the controllable protein sequence generation task, ProLLaMA can **design novel proteins with desired functionalities**. In the protein property prediction task, ProLLaMA achieves nearly 100\% accuracy across many categories. The latter two tasks are beyond the reach of other ProLLMs

<p align="center"><img src="img/intro.png" title="" height="500"></p>

<details open><summary> **I also have other AI for Science projects that may interest you.** </summary><p>
<!--  may -->

> [TaxDiff: Taxonomic-Guided Diffusion Model for Protein Sequence Generation](https://arxiv.org/abs/2402.17156) [![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/PKU-YuanGroup/TaxDiff) <br><br>
> [Navigating Chemical-Linguistic Sharing Space with Heterogeneous Molecular Encoding](https://arxiv.org/abs/2412.20888) [![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/Lyu6PosHao/HME) <br><br>
> DM-Assembler: Leveraging Domain Motif Assembler for Multi-objective, Multi-domain and Explainable Molecular Design [![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/cziun/DM-Assembler) <br><br>

</p ></details>

## 💡Highlights
### Powerful model
* Our ProLLaMA is the first model to our knowledge capable of simultaneously handling multiple PLP tasks.
* **including generating proteins with specified functions based on the user's intent.**

### General training framework
* We propose a training framework with scalability and efficiency that enables any general LLM to be trained as a proficient model for multiple tasks in Protein Language Processing.

### Excellent performance
* Experiments show that our ProLLaMA not only handles PLP tasks beyond the reach of existing ProLLMs but also achieves state-of-the-art results in the protein generation task where current ProLLMs are active.

## 😮Main Results
* Protein sequence generation
  <p align="center"><img src="img/r4.jpg" title=""></p>
  <p align="center"><img src="img/r1.jpg" title=""></p>
  
* Controllable protein sequence generation (controlled by the given [superfamily descriptions](https://github.com/Lyu6PosHao/ProLLaMA/blob/main/superfamilies.txt))
  
  1 cases for each superfamily: ProLLaMA is capable of generating desired proteins (Blue) with functions and structures similar to natural proteins (Yellow).
  <p align="center"><img src="img/r5.png" title=""></p>
  100 cases for each superfamily:
  <p align="center"><img src="img/r2.jpg" title=""></p>
  
* Protein property prediction
  
  <p align="center"><img src="img/r3.jpg" title="" ></p>

## 🚀Pipeline
The training framework we propose is as follows:
* (A) Continual learning on protein language.
* (B) Instruction tuning on multi-tasks.
* (C) Expanding to more tasks by instruction tuning in the future.
<p align="center"><img src="img/train_framework_v3.png" title=""></p>

## 🛠️Quick Inference
**As ProLLaMA's architecture is the same as LLaMA2, you can use ProLLaMA for inference like using LLaMA2.**

Follow the steps below to use our ProLLaMA for inference.
### 1.Install Requirements

* torch==2.0.1
* transformers==4.35.0
* cuda==11.7
```bash
git clone https://github.com/Lyu6PosHao/ProLLaMA.git
cd ProLLaMA
pip install -r requirements.txt
```

### 2.Download Model
Download from [Hugging Face](https://huggingface.co/GreatCaptainNemo/ProLLaMA)

### 3.Usage

**Just like using LLaMA2, three ways are provided here:**

* Commandline

```bash
CUDA_VISIBLE_DEVICES=0 python ./scripts/infer.py --model "GreatCaptainNemo/ProLLaMA" --interactive
#You can replace the model_path with your local path
#Make sure you use only one GPU for inference
#Use "python ./scripts/infer.py -h" for more details
```

* Python
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
from tqdm import tqdm
device=torch.device('cuda:0')

##You can replace the file_path with your local path
tokenizer = AutoTokenizer.from_pretrained("GreatCaptainNemo/ProLLaMA", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("GreatCaptainNemo/ProLLaMA", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
generation_config = GenerationConfig(temperature=0.2,top_k=40, top_p=0.9,do_sample=True,num_beams=1,repetition_penalty=1.2,max_new_tokens=400)
model.eval()
print("####Enter 'exit' to exit.")
with torch.no_grad():
    while True:
        messages = []
        user=str(input("Input:"))
        if user.strip()=="exit":
            break
        inputs = tokenizer(user, return_tensors="pt").to(device)
        generate_ids = model.generate(inputs.input_ids,generation_config).to(device)
        response=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print("Output:", response)
```

* [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
python ./src/cli_demo.py \
      --model_name_or_path /path_to_your_model \
      --template llama2
```

### 4.Input Format
The instructions which you input to the model should follow the following format:
```text
[Generate by superfamily] Superfamily=<xxx>
or
[Determine superfamily] Seq=<yyy>
```
Here are some examples of the input:
```text
[Generate by superfamily] Superfamily=<Ankyrin repeat-containing domain superfamily>
```
```
#You can also specify the first few amino acids of the protein sequence:
[Generate by superfamily] Superfamily=<Ankyrin repeat-containing domain superfamily> Seq=<MKRVL
```
```
[Determine superfamily] Seq=<MAPGGMPREFPSFVRTLPEADLGYPALRGWVLQGERGCVLYWEAVTEVALPEHCHAECWGVVVDGRMELMVDGYTRVYTRGDLYVVPPQARHRARVFPGFRGVEHLSDPDLLPVRKR>
```
**See [this](https://github.com/Lyu6PosHao/ProLLaMA/blob/main/superfamilies.txt) on all the optional superfamilies.**

## 🛠️Quick Train
### Stage 1
1. Prepare the dataset: put your dataset under **./scripts/pretraining_dataset**. You dataset should be one or several **txt files**. Each line in the txt file should be **one protein sequence** in the format of "Seq=<xxx>". We provided ./scripts/pretraining_dataset/example.txt as an example.
2. Run ./scripts/run_pt.sh
### Stage 2
1. Prepare the dataset: download our instruction_dataset from [HuggingFace](https://huggingface.co/datasets/GreatCaptainNemo/instruction_dataset) and put the train_split under ./scripts/instruction_tuning_dataset. We provided ./scripts/instruction_tuning_dataset/example.json as an example.
2. Run ./scripts/run_it.sh
3. If you want to fine-tune our ProLLaMA on your own dataset instead of our instruction_dataset, you should process your data **into the similar format** like our instruction_dataset (or example.json).
4. It may be better to fine-tune **ProLLaMA_Stage_1** instead of ProLLaMA if your dataset is relatively small and not relevant to superfamily tasks.
## ✒️Others
### ProLLaMA of Stage 1

ProLLaMA_Stage_1 refers to the model obtained by continual pre-training LLaMA2 on the UniRef50 dataset, as shown in the [pipeline](https://github.com/PKU-YuanGroup/ProLLaMA?tab=readme-ov-file#pipeline). [Model Weights](https://huggingface.co/GreatCaptainNemo/ProLLaMA_Stage_1)

You can use ProLLaMA_Stage_1 in the same way as ProLLaMA. For example:
```bash
CUDA_VISIBLE_DEVICES=0 python ./scripts/infer.py --model "GreatCaptainNemo/ProLLaMA_Stage_1" --interactive
#You can replace the model_path with your local path
#Make sure you use only one GPU for inference
#Use "python ./scripts/infer.py -h" for more details
```

However, ProLLaMA_Stage_1's input format is a little different from ProLLaMA, since the former is only trained on pure protein sequences without nautral language instructions.

The input format:
```text
Seq=
#You can also specify the first few amino acids of the protein sequence:
Seq=<MAPGGMPRE
```
You can perform instruction tuning on ProLLaMA_Stage_1 (or ProLLaMA) with your custom datasets, in order to make the model capable of your insterested PLP tasks.

We plan to build a more powerful ProLLaMA_Stage_1.

## Logs
[2025-01-07]
- The peft codes in the **src/peft** is not used. The directory has been renamed to **src/peft(deprecated)**.
- The checkpoints during training will be saved in ${output_dir}. And when **"merge_when_finished"** is True, the LoRA adapters will be merged into the base model, and the merged model will be saved in  ${output_dir}_merged. Then you can easily use transformers.AutoModelForCausalLM.from_pretrained() to load the merged model directly.

## ✏️Citation
If you find our repo helpful, please consider citing us.
```BibTex
@article{lv2024prollama,
  title={ProLLaMA: A Protein Large Language Model for Multi-Task Protein Language Processing},
  author={Lv, Liuzhenghao and Lin, Zongying and Li, Hao and Liu, Yuyang and Cui, Jiaxi and Chen, Calvin Yu-Chian and Yuan, Li and Tian, Yonghong},
  journal={arXiv preprint arXiv:2402.16445},
  year={2024}
}
```
