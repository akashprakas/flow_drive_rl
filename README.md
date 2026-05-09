

# NavSim Evaluation Results

Eval split: `navmini` | Metric: PDMS (PDM Score)

## Results Summary

| Agent | Checkpoint | N | Mean PDMS | Median |
|-------|-----------|---|-----------|--------|
| DiffusionDrive | `diffusiondrive_navsim_88p1_PDMS` | 392 | **0.8235** | 0.8524 |
| FlowAgent (warm-start) | `diffusiondrive_navsim_88p1_PDMS` | 392 | **0.8255** | 0.8534 |
| Constant Velocity | — | 396 | 0.3702 | 0.0000 |

> Note: The 88.1 PDMS reported in the DiffusionDrive paper is on `navhard_two_stage`, not `navmini`.

---

## Eval Commands

### DiffusionDrive (pretrained)

```bash
source setup_env.sh && python navsim/planning/script/run_pdm_score.py \
    agent=diffusiondrive_agent \
    train_test_split=navmini \
    experiment_name=diffusiondrive_pretrained_navmini \
    metric_cache_path=$(pwd)/exp/metric_cache_mini \
    agent.checkpoint_path=$(pwd)/ckpts/diffusiondrive_navsim_88p1_PDMS/diffusiondrive_navsim_88p1_PDMS \
    worker.threads_per_node=4
```

### FlowAgent (warm-started from DiffusionDrive checkpoint)

```bash
source setup_env.sh && python navsim/planning/script/run_pdm_score.py \
    agent=flow_agent \
    train_test_split=navmini \
    experiment_name=flow_agent_pretrained_eval \
    metric_cache_path=$(pwd)/exp/metric_cache_mini \
    agent.checkpoint_path=$(pwd)/ckpts/diffusiondrive_navsim_88p1_PDMS/diffusiondrive_navsim_88p1_PDMS \
    worker.threads_per_node=4
```

### Constant Velocity (baseline)

```bash
source setup_env.sh && python navsim/planning/script/run_pdm_score.py \
    agent=constant_velocity_agent \
    train_test_split=navmini \
    experiment_name=constant_velocity_navmini \
    metric_cache_path=$(pwd)/exp/metric_cache_mini \
    worker.threads_per_node=4
```




source setup_env.sh && python navsim/planning/script/run_training.py \
    agent=flow_agent \
    train_test_split=navmini \
    experiment_name=flow_agent_navmini_train \
    cache_path=$(pwd)/exp/training_cache_mini \
    force_cache_computation=false \
    use_cache_without_dataset=false \
    trainer.params.max_epochs=20 \
    trainer.params.accelerator=gpu \
    trainer.params.strategy=auto \
    trainer.params.precision=32 \
    dataloader.params.batch_size=16 \
    dataloader.params.num_workers=2