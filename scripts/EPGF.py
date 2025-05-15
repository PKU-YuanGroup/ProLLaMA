# protein_sequence_scorer.py

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter
import re
import math
import random
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig
from tqdm import tqdm

class ProteinSequenceScorer:
    """
    A protein sequence scoring utility that evaluates amino acid sequences across multiple criteria:
    - Basic composition
    - Physicochemical properties
    - Sequence complexity
    - Functional motifs
    """
    
    def __init__(self, sequence):
        self.sequence = sequence.upper()
        self.clean_sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', self.sequence)
        self.analyzer = ProteinAnalysis(self.clean_sequence)

        self.hydrophobicity = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }

        self.aa_groups = {
            'Hydrophobic': ['A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V'],
            'Hydrophilic': ['R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'K', 'P', 'S', 'T'],
            'Positive': ['H', 'K', 'R'],
            'Negative': ['D', 'E'],
            'Polar': ['N', 'Q', 'S', 'T', 'Y'],
            'Aromatic': ['F', 'W', 'Y'],
            'Small': ['A', 'G', 'S']
        }

    def _calculate_distribution_score(self, distribution):
        values = list(distribution.values())
        total = sum(values)
        probabilities = [v / total for v in values]
        entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probabilities)
        max_entropy = math.log2(len(values)) if len(values) > 0 else 1
        return entropy / max_entropy if max_entropy > 0 else 0

    def score_basic_composition(self):
        result = {}
        composition = self.analyzer.get_amino_acids_percent()
        result['AA_Distribution_Score'] = self._calculate_distribution_score(composition)

        unique_aa_count = len(set(self.clean_sequence))
        result['AA_Diversity'] = unique_aa_count / 20

        rare_aa = ['C', 'H', 'M', 'W']
        rare_count = sum(1 for aa in self.clean_sequence if aa in rare_aa)
        result['Rare_AA_Ratio'] = rare_count / len(self.clean_sequence) if self.clean_sequence else 0

        return result

    def score_physicochemical(self):
        result = {}
        hydro_values = [self.hydrophobicity.get(aa, 0) for aa in self.clean_sequence]
        result['Mean_Hydrophobicity'] = sum(hydro_values) / len(hydro_values) if hydro_values else 0

        min_hydro, max_hydro = -4.5, 4.5
        result['Normalized_Hydrophobicity'] = (result['Mean_Hydrophobicity'] - min_hydro) / (max_hydro - min_hydro)

        for group_name, group_aas in self.aa_groups.items():
            count = sum(1 for aa in self.clean_sequence if aa in group_aas)
            result[f'{group_name}_Ratio'] = count / len(self.clean_sequence) if self.clean_sequence else 0

        positive = result['Positive_Ratio']
        negative = result['Negative_Ratio']
        result['Charge_Balance'] = 1 - abs(positive - negative)

        result['Instability_Index'] = self.analyzer.instability_index()
        result['Stability_Score'] = 1 - min(result['Instability_Index'] / 100, 1)

        return result

    def score_complexity(self):
        result = {}
        aa_counter = Counter(self.clean_sequence)
        total = len(self.clean_sequence)
        entropy = -sum((count / total) * math.log2(count / total) for count in aa_counter.values())
        max_entropy = math.log2(min(20, total))
        result['Sequence_Entropy'] = entropy / max_entropy if max_entropy > 0 else 0

        repeats_score = 0
        for k in range(2, 6):
            if len(self.clean_sequence) < k:
                continue
            patterns = {}
            for i in range(len(self.clean_sequence) - k + 1):
                pattern = self.clean_sequence[i:i + k]
                patterns[pattern] = patterns.get(pattern, 0) + 1
            repeats = sum(count - 1 for count in patterns.values() if count > 1)
            max_repeats = len(self.clean_sequence) - k
            repeats_score += repeats / max_repeats if max_repeats > 0 else 0

        result['Repeat_Pattern_Score'] = 1 - (repeats_score / 4) if repeats_score <= 4 else 0

        window_size = 20
        complexity_scores = []
        for i in range(0, len(self.clean_sequence) - window_size + 1, 10):
            window = self.clean_sequence[i:i + window_size]
            unique_aa = len(set(window))
            complexity_scores.append(unique_aa / window_size)

        result['Local_Complexity_Mean'] = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0

        return result

    def score_functional(self):
        result = {}
        ss = self.analyzer.secondary_structure_fraction()
        result['Helix_Tendency'] = ss[0]
        result['Sheet_Tendency'] = ss[1]
        result['Coil_Tendency'] = ss[2]

        motifs = {
            'N_Glycosylation': r'N[^P][ST][^P]',
            'Phosphorylation': r'[ST]P',
            'DNA_Binding': r'[KR]{3,}',
            'Zinc_Finger': r'C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H',
            'Nuclear_Localization': r'[KR]{4,}',
        }

        for motif_name, pattern in motifs.items():
            matches = len(re.findall(pattern, self.clean_sequence))
            result[f'{motif_name}_Count'] = matches
            result[f'{motif_name}_Density'] = matches / (len(self.clean_sequence) / 100) if self.clean_sequence else 0

        mw = self.analyzer.molecular_weight()
        result['Molecular_Weight'] = mw
        result['MW_Score'] = min(mw / 100000, 1) if mw > 0 else 0

        return result

    def get_comprehensive_score(self):
        composition = self.score_basic_composition()
        physicochemical = self.score_physicochemical()
        complexity = self.score_complexity()
        functional = self.score_functional()

        key_scores = [
            composition['AA_Distribution_Score'],
            composition['AA_Diversity'],
            physicochemical['Normalized_Hydrophobicity'],
            physicochemical['Stability_Score'],
            complexity['Sequence_Entropy'],
            complexity['Local_Complexity_Mean'],
            complexity['Repeat_Pattern_Score'],
            (functional['Helix_Tendency'] + functional['Sheet_Tendency']) / 2,
        ]

        comprehensive_score = sum(key_scores) / len(key_scores)
        return comprehensive_score

    def get_all_scores(self):
        return {
            "Comprehensive_Score": self.get_comprehensive_score(),
            "Basic_Composition_Score": self.score_basic_composition(),
            "Physicochemical_Score": self.score_physicochemical(),
            "Sequence_Complexity_Score": self.score_complexity(),
            "Functional_Score": self.score_functional()
        }


# === Evolutionary Protein Generation Framework (EPGF) with ProLLaMA === #

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

model_path = "GreatCaptainNemo/ProLLaMA"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path, device_map='cuda:0', torch_dtype=torch.bfloat16)

generation_config = GenerationConfig(
    max_new_tokens=1,
    do_sample=True,
    top_k=40,
    top_p=0.9,
    temperature=1,
    num_return_sequences=8,
    repetition_penalty=1,
)

def softmax(x, temperature=1.0):
    x = np.array(x) / temperature
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

initial_temperature = 1.0
final_temperature = 0.001
decay_rate = 0.1

output_file = "./output_of_prollama_CheY-like.fasta"
model.eval()

with torch.no_grad():
    i = 0
    #gennerate 100 protein sequences
    while i < 100:
        temperature = initial_temperature
        failed = False
        prompt = '[Generate by superfamily] Superfamily=<CheY-like superfamily> Seq=<'  #you can modify this prompt

        while True:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                output_scores=True,
                return_dict_in_generate=True
            )

            candidates = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
            log_probs = transition_scores.sum(dim=1).cpu().numpy().tolist()

            candidates_with_scores = [(candidates[idx], log_probs[idx]) for idx in range(len(candidates))]
            candidates_with_scores.sort(key=lambda x: x[1], reverse=True)

            top_half_count = max(1, len(candidates) // 2)
            top_candidates = candidates_with_scores[:top_half_count]

            for cand, score in candidates_with_scores:
                if cand.strip().endswith('>') and (cand, score) not in top_candidates:
                    top_candidates.append((cand, score))

            bio_scores = []
            filtered_candidates = []

            for cand, lm_score in top_candidates:
                seq = cand.split('Seq=<')[-1].split('>')[0].strip()
                scorer = ProteinSequenceScorer(seq)
                bio_score = scorer.get_comprehensive_score()

                if bio_score < 0.55:
                    continue
                bio_scores.append(bio_score)
                filtered_candidates.append(cand)

            if len(filtered_candidates) == 0:
                failed = True
                break

            bio_scores_softmax = softmax(bio_scores, temperature=temperature)
            temperature = max(final_temperature, temperature * decay_rate)
            sampled_idx = np.random.choice(len(filtered_candidates), p=bio_scores_softmax)
            winner = filtered_candidates[sampled_idx]

            if winner.endswith('>'):
                break
            else:
                prompt = winner

        if not failed:
            with open(output_file, 'a') as f:
                f.write(f'>{i}\n')
                f.write(winner.split('Seq=<')[-1].split('>')[0].strip() + '\n')
                i += 1
