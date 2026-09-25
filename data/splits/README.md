# The paper's UniRef50 / InterPro split

These four files are the exact data the base SAE (`models/sae_k32_time_layer_m1.pt`) was trained and
evaluated on. They are committed because **re-deriving them does not reproduce them.**

| File | Bytes | SHA-256 (first 16) |
| --- | --- | --- |
| `uniref50_interpro_train_1000.fasta` | 302121 | `02fd758adafeb5d7` |
| `uniref50_interpro_eval_1000.fasta` | 307216 | `025afed8ce1b58df` |
| `interpro_train_1000.jsonl` | 660943 | `04cc1e75e937b04e` |
| `interpro_eval_1000.jsonl` | 697103 | `dee5d2349eb44067` |

## Why they are committed rather than regenerated

These were built from UniRef50 plus the InterPro API. Rebuilding them today produces a *different*
1000-sequence split, for two reasons that no seed can control:

1. **The UniRef50 release is unpinned.** `uniref50.fasta.gz` is replaced roughly every 8 weeks, and
   the reservoir sampler is deterministic given an identical byte stream and nothing else.
2. **InterPro is queried live.** A candidate is kept only if the API returns spans for it, so both
   membership and ordering shift with each InterPro release.

The seed (`uniref50.seed: 0`) therefore reproduces the *procedure*, not the *dataset*. So the files
ship: they are what the released SAE was trained on, and they are what `confsteer sae-encode` reads
by default.

Sequence identity is the statistical unit throughout: the train and eval splits share no sequence.
