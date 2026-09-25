# noisy-mastermind

Example runs:

```
python run.py --reasoning-effort max  -m gpt-5.6-terra  --n_samples 40 --seed 857 --note "gpt-5.6-terra-max"
python run.py --reasoning-effort low  -m gpt-5.6-terra  --n_samples 40 --seed 857 --note "gpt-5.6-terra-low"
```

Distributed local runs:
```
python run_queue_8192.py
...
```