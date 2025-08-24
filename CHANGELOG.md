# CHANGELOG

## 24 Aug 2025

1. Fixed Types format, added Defaults for Convolution block.
2. Added Convolution block: Conv1d --> BatchNorm --> activation --> dropout.

## 22 Aug 2025

1. Added python script featureExtract.py where:
    - Mode finetune creates a dnaber6 model and finetunes it with dataset given.
    - Mode features loads a dnabert6 model from the given path and extracts embeddings from the model, computes other features and converts sequences to 1hot encoded.
      Then saves the results in a pt file. With keys: ["sequences", "labels", "onehot", "embeddings"].
2. Added more Default values for DNABERT6 model
3. Fixed command for featureExtract features mode to be more realistic.

## 15 Aug 2025

1. Minor code fixes in dnabert6 file, removed test code.
2. Added more default for dnabert6 into Types.

## 9 Aug 2025

1. Changed metrics function in DNABERT6, in order to:
    - Account for the type EvalPrediction that Hugging Face's Trainer passes in metrics function.
    - EvalPrediction inclued at least the fields below:
        1. predictions: model outputs for each example (for classification, these are logits with shape (N, num_labels)).
        2. label_ids: the true labels with shape (N,).
        3. In some setups, models return tuples (e.g., extra outputs like hidden states). 
           That can bubble up so evalPred.predictions becomes something like (logits, ) or (logits, extra). 
           This case is handled with the code below

            ```python
            logits = evalPred.predictions[0] if isinstance(evalPred.predictions, tuple) else evalPred.predictions
            ```
    - For DNABERT classification, almost always we get a plain NumPy array of logits (N, 2)
      but that tuple-guard prevents rare crashes.

## 9 Aug 2025

1. Removed padding from dataread, as dnabert tokenizer does not recognize N as PAD characters.
2. Keep N only for ambiguous characters, then replace them with MASK.
3. dnabert tokenizer does not create 6-mer sequences from a consecutive DNA sequence, we need to pass 6-mers to dnabert tokenizer.
4. Added  kmer function to create kmers for dnabert input.
5. Fixed encoder function to first create the batched kmers then return tokenizer, same logic for embeddings function.
6. Added new state KmerAmbiguousState, which decides what token will be used for ambiguous kmers in kmer function ([MASK] or [UNK]).