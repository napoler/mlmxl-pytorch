## MLM (Masked Language Modeling Memory Transformer-XL) Pytorch

This repository allows you to quickly setup unsupervised training for your transformer off a corpus of sequence data.

## Install

```bash
$ pip install git+https://github.com/napoler/mlmxl-pytorch
```

## Usage

First `pip install reformer-pytorch`, then run the following example to see what one iteration of the unsupervised training is like

```python
import torch
from memory_transformer_xl import MemoryTransformerXL
import torch
from torch import nn
from torch.optim import Adam
# from mlm_pytorch import MLM

# instantiate the language model


from transformers import AutoTokenizer, AutoModel,BertTokenizer
  
tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_clue_tiny")

model = MemoryTransformerXL(
    num_tokens = tokenizer.vocab_size,
    dim = 128,
    heads = 8,
    depth = 8,
    seq_len = 1024,
    mem_len = 256,            # short term memory (the memory from transformer-xl)
    lmem_len = 256,           # long term memory (memory attention network attending to short term memory and hidden activations)
    mem_write_iters = 2,      # number of iterations of attention for writing to memory
    memory_layers = [6,7,8],  # which layers to use memory, only the later layers are actually needed
    num_mem_kv = 128,         # number of memory key/values, from All-attention paper

).cuda()

x1 = torch.randint(0, tokenizer.vocab_size, (1, 1024)).cuda()
logits1, mem1 = model(x1)

x2 = torch.randint(0, tokenizer.vocab_size, (1, 1024)).cuda()
logits2, mem2 = model(x2, memories = mem1)
# plugin the language model into the MLM trainer

# plugin the language model into the MLM trainer

trainer = MLMXL(
    model,
    mask_token_id = tokenizer.mask_token_id,          # the token id reserved for masking
    pad_token_id = tokenizer.pad_token_id,           # the token id for padding
    mask_prob = 0.15,           # masking probability for masked language modeling
    replace_prob = 0.90,        # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
    mask_ignore_token_ids = [tokenizer.cls_token_id,tokenizer.sep_token_id]  # other tokens to exclude from masking, include the [cls] and [sep] here
).cuda()

# optimizer

opt = Adam(trainer.parameters(), lr=3e-4)

# one training step (do this for many steps in a for loop, getting new `data` each time)

data = torch.randint(0, tokenizer.vocab_size, (2, 1024)).cuda()

loss = trainer(data)
loss.backward()
opt.step()
opt.zero_grad()

# after much training, the model should have improved for downstream tasks

# torch.save(transformer, f'./pretrained-model.pt')

# after much training, the model should have improved for downstream tasks

torch.save(transformer, f'./pretrained-model.pt')
```

Do the above for many steps, and your model should improve.

## Citation

```bibtex
@misc{devlin2018bert,
    title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
    author={Jacob Devlin and Ming-Wei Chang and Kenton Lee and Kristina Toutanova},
    year={2018},
    eprint={1810.04805},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
