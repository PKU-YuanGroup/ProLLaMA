**mutation.py** is used to measure mutation effects. Note that this script is exemplary only and does not guarantee academic rigor. You may need to modify it.

We calculate neg_log_likelihood (nll in short) of wild_type and mutated_type respectively. And then we calculated the fitness score according to [1].

The code is based on [this](https://huggingface.co/docs/transformers/v4.39.3/en/perplexity#perplexity-of-fixed-length-models).

**Reference:**

[1] Tranception: Protein Fitness Prediction with Autoregressive Transformers and Inference-time Retrieval

**Tips:**
- The sequence can be a pure protein sequence or a sequence with the instruction. For example:
```
#pure protein sequence
Seq=<MAPGGMPREFPSFVRTLPEADLGYPALRGWVLQGERGCVLYWEAVTEVALPEHCHAECWGVVVDGRMELMVDGYTRVYTRGDLYVVPPQARHRARVFPGFRGVEHLSDPDLLPVRKR>

#sequence containing instructions
[Generate by superfamily] Superfamily=<RmlC-like cupin domain superfamily> Seq=<MAPGGMPREFPSFVRTLPEADLGYPALRGWVLQGERGCVLYWEAVTEVALPEHCHAECWGVVVDGRMELMVDGYTRVYTRGDLYVVPPQARHRARVFPGFRGVEHLSDPDLLPVRKR>
```

- Maybe the score can be normalized according to the length.
