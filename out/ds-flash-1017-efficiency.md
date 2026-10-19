# deepseek-flash → DeepSeek-V4.1-Flash-1017: reasoning efficiency

Endgames60 benchmark: the same 60 positions, each played once at batch sizes 2, 3 and 4 (180 games per model), reasoning effort `low`.

| Batch size | Solved (old → new) | Δ accuracy | Tokens/solve (old → new) | Δ tokens/solve |
|---|---|---|---|---|
| 2 | 88% → 97% | +8 pp [−2, +18] | 42.7k → 30.8k | **−28% [−43, −10]** |
| 3 | 80% → 93% | +13 pp [0, +27] | 35.5k → 25.6k | **−28% [−43, −11]** |
| 4 | 77% → 83% | +7 pp [−8, +22] | 30.7k → 24.8k | **−19% [−36, −1]** |
| **All** | **82% → 91%** | **+9 pp [+3, +16]** | **36.6k → 27.2k** | **−26% [−35, −15]** |

Brackets are 95% bootstrap confidence intervals (10,000 resamples). **Bold** means the interval excludes zero. The size-3 accuracy interval's lower bound is 0, so it is left unbolded.

- Accuracy CIs resample the 60 positions, because each position appears in all three batch sizes.
- Token CIs resample the paired batches; both models ran identical batches.
- Tokens/solve = total output tokens / games solved.
- 4 runs hit the 131,072-token output cap (3 old, 1 new) and count at the cap. This favors the old model slightly.
