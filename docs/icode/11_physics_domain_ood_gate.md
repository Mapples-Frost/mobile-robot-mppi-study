# Physics-Domain and OOD Gate

## Question

Does a residual model trained on several MuJoCo physics domains improve
prediction and MPPI control when mass, inertia, friction, command delay and
actuator strength change together in a held-out combination?

## Domain design

Seen domains:

1. nominal physics;
2. 11 kg chassis with increased inertia;
3. reduced wheel and floor friction;
4. 80 ms command delay.

Held-out domain:

```text
12 kg chassis
increased inertia
100 ms command delay
lower wheel/floor friction
1.8 torque limit
reduced actuator kp
```

Each seen domain is split by complete episode and contributes independent
episodes to train, validation and test. The combined held-out domain contributes
only to `unseen`; none of its transitions influence normalization, optimization,
early stopping or checkpoint selection.

## Executed smoke dataset (2026-07-12)

```text
domains:                         5
episodes per domain:             6
transitions per episode:        30
total episodes:                 30
total transitions:             900
train / validation / test: 16 / 4 / 4 seen episodes
unseen:                          6 episodes
residual identity max error:     0
transition closure max error:    3.32e-17
command/applied mismatch:      100%
```

## Twenty-epoch prediction smoke

Seen test:

| Method | Residual RMSE | H=1 | H=5 | H=10 | H=20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nominal | 0.46928 | -- | -- | -- | 0.04427 |
| MLP | 0.39024 | 0.02461 | 0.02786 | 0.02816 | 0.02939 |
| ICODE | 0.32488 | 0.02275 | 0.02575 | 0.02697 | 0.02890 |

Combined unseen test:

| Method | Residual RMSE | H=1 | H=5 | H=10 | H=20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nominal | 0.65046 | -- | -- | -- | 0.05629 |
| MLP | 0.59898 | 0.03497 | 0.04359 | 0.04521 | 0.04727 |
| ICODE | 0.51707 | 0.03168 | 0.04011 | 0.04303 | 0.04601 |

ICODE improves the held-out prediction, but the OOD margin is smaller than the
seen-domain margin. This is expected and must remain visible in reporting.

## Three-seed held-out-physics control smoke

The held-out physics was combined with the clean single-obstacle navigation
scene. All methods used 400 samples, the same seeds and the same safety chain.

| Method | Success | Collision | Mean final distance | Mean safety interventions |
| --- | ---: | ---: | ---: | ---: |
| Nominal MPPI | 0/3 | 1/3 | 1.293 m | 171.7 |
| MLP-MPPI | 1/3 | 0/3 | 1.873 m | 175.3 |
| ICODE-MPPI | 2/3 | 0/3 | 0.386 m | 27.0 |

ICODE planning averaged 57.2 ms. Two isolated deadline misses occurred across
the three complete episodes; therefore the real-time gate is promising but not
yet perfect.

These three seeds are a smoke result, not a publishable success-rate estimate.
The nominal collision also shows that `scan_guard` cannot mathematically
guarantee collision avoidance on low-friction, delayed plants; it retains final
command authority but cannot remove braking-distance physics.

## CPU inference decision

The online model uses 64x64 hidden layers and one Torch thread. TorchScript is
enabled for rollout inference, while a NumPy-boundary finite check remains.

```text
600 samples: mean 81.8 ms, 2/30 deadline misses
400 samples: mean 57.6 ms, 0/30 deadline misses in timing smoke
```

The current CPU working point is 400 samples. Six hundred samples remains a GPU
or compute-performance ablation.

## Required next evidence

- expand to at least 10 matched control seeds, then 20 for the main table;
- train with several dataset and network initialization seeds;
- increase each domain from smoke scale to thousands of transitions;
- report per-domain seen results, not only pooled test metrics;
- repeat in `lab_complex`, `narrow_corridor` and `u_trap_long_board`;
- keep dynamic obstacles for the later RL-prior study, so the ICODE claim stays
  about dynamics mismatch rather than obstacle prediction.
