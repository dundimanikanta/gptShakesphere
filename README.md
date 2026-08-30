# GPT Shakespeare

Following Andrej Karpathy's "Let's build GPT" lecture, training a character-level language model on Shakespeare text.

## Training progress

Validation/train loss at step 49500 as features are added incrementally:

| Stage | train loss | val loss |
|---|---|---|
| Initial bigram (no embeddings) | 2.4561 | 2.5054 |
| Bigram + embeddings | 2.5222 | 2.5472 |
| + single self-attention head | 2.4822 | 2.5440 |
| + multi-headed self-attention | 2.1834 | 2.2122 |
| + feedforward | 2.1099 | 2.1911 |
| + adding blocks | 2.1261 | 2.2355 |
