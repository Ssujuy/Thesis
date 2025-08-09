# CHANGELOG

## 9 Aug 2025

1. Removed padding from dataread, as dnabert tokenizer does not recognize N as PAD characters.
2. Keep N only for ambiguous characters, then replace them with MASK.
3. dnabert tokenizer does not create 6-mer sequences from a consecutive DNA sequence, we need to pass 6-mers to dnabert tokenizer.
4. Added  kmer function to create kmers for dnabert input.
5. Fixed encoder function to first create the batched kmers then return tokenizer, same logic for embeddings function.
6. Added new state KmerAmbiguousState, which decides what token will be used for ambiguous kmers in kmer function ([MASK] or [UNK]).