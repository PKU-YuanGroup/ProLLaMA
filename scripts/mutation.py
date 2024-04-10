import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

#the smaller the nll is, the better
def neg_log_likelihood(model, tokenizer,sequence, device):
    input_ids = tokenizer(sequence, return_tensors="pt").input_ids.to(device)
    target_ids = input_ids.clone()
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        nll= outputs.loss
    return nll

if __name__ == '__main__':
    device = torch.device('cuda:0')
    #change "path_to_the_model" to your valid path
    tokenizer = AutoTokenizer.from_pretrained("path_to_the_model", use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("path_to_the_model", device_map=device, torch_dtype=torch.bfloat16, trust_remote_code=True)
    wild_type='Seq=<MAPGGMPREFPSFVRTLPEADLGYPALRGWVLQGERGCVLYWEAVTEVALPEHCHAECWGVVVDGRMELMVDGYTRVYTRGDLYVVPPQARHRARVFPGFRGVEHLSDPDLLPVRKR>'
    mutated_type='Seq=<AAPGGMPREFPSFVRTLPEADLGYPALRGWVLQGERGCVLYWEAVTEVALPEHCHAECWGVVVDGRMELMVDGYTRVYTRGDLYVVPPQARHRARVFPGFRGVEHLSDPDLLPVRKR>'
    nll_wt = neg_log_likelihood(model, tokenizer, wild_type, device).item()
    nll_mt = neg_log_likelihood(model, tokenizer, mutated_type, device).item()

    fitness_score=-nll_mt+nll_wt
    print('nll_wt:', nll_wt)
    print('nll_mt:', nll_mt)
    print(f'Fitness score of the mutated sequence is: {fitness_score}')
