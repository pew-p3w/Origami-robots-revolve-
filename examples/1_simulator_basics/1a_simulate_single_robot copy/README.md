# 1a Simulate Single Robot

This folder contains the ball-approach evolutionary experiment. The experiment
trains a modular robot controller to move toward a ball, using fixed training
ball positions per run and random ball positions for visual testing.

The repository-level `run.py` is the intended entry point. It loads a selected
configuration file, runs this folder's `main.py`, and writes artifacts under
`output/`.

## Repository Layout

```text
revolve2-folding/
├── run.py
├── config/
│   └── 1_examples/
│       ├── <body_config>.py
│       └── ...
├── examples/
│   └── 1_simulator_basics/
│       └── 1a_simulate_single_robot/
│           ├── README.md
│           ├── ball_aware_brain.py
│           ├── evaluator.py
│           ├── genotype.py
│           ├── main.py
│           ├── requirements.txt
│           └── test_best.py
├── jobs/
│   ├── <body>.job
│   └── logs/
└── output/
    └── <run_name>/
```

## Run Commands

Training:

```bash
python run.py -r config/1_examples/<body_config>.py examples/1_simulator_basics/1a_simulate_single_robot output/<run_name>
```

Continue from a generation snapshot:

```bash
python run.py -c output/<run_name>/gen<N>.pkl
```

Test one saved generation visually:

```bash
python run.py -t output/<run_name>/gen<N>.pkl
```

Preview a body with one random controller:

```bash
python run.py --random-test config/1_examples/<body_config>.py
```

Export one CSV from all generation snapshots:

```bash
python run.py -o output/<run_name>
```

## Configuration Files

Configuration files live under `config/1_examples/`. Each body has its own
config file, but the structure is intentionally the same:

- `BODY`: the robot body used for the run.
- Current configs cover the custom `simple` body plus the standard v2 bodies:
  `gecko`, `spider`, `ant`, and `snake`.
- Terrain and ball settings: terrain size, ball radius/mass, spawn margin, and
  reached-ball distance.
- Training ball sampling: `NUM_TRAINING_BALL_POSES` and
  `MIN_TRAINING_BALL_DISTANCE_FRACTION`.
- Brain feedback settings: live ball-position feedback input count, output
  scale, and distance scale.
- EA settings: population size, tournament size, number of generations,
  mutation standard deviation, and per-gene mutation probability.
- Simulation settings: simulation time, number of parallel simulators, and
  headless mode.
- Helper functions: `parameter_filename()`, `make_terrain()`,
  `make_random_ball_pose()`, and `make_training_ball_pose()`.

`make_random_ball_pose()` is used for testing, so test scenes remain random.
`make_training_ball_pose()` is used for training and rejects balls that are too
close to the robot's starting point.

## Experiment Files

`main.py`  
Main evolutionary algorithm. It loads the active config through `run.py`, creates
the evaluator, initializes the population, runs the generational GA, writes CSV
rows, saves generation snapshots, and supports resume from snapshots.

`evaluator.py`  
Builds one MuJoCo scene per genotype for each fixed training ball position. It
adds the robot, ball, terrain, and reached-ball stop condition, then computes the
normalized fitness for each controller.

`ball_aware_brain.py`  
Defines the CPG controller with live ball-position feedback. The genotype is
split into CPG connection weights plus steering-layer weights. During simulation,
the brain reads the current robot-to-ball vector and adds evolved steering
offsets to the CPG outputs.

`genotype.py`  
Defines the flat parameter vector used by the EA. It provides random
initialization, copying, one-point crossover that returns two children, and
per-gene Gaussian mutation with clipping to `[-1.0, 1.0]`.

`test_best.py`  
Legacy visual test script for this folder. The preferred visual test path is now
`run.py -t`, because generation snapshots include the config source and code
paths needed to reconstruct the run environment.

`requirements.txt`  
Placeholder for example-specific dependencies.

## Fitness

Each controller is evaluated on the fixed training ball positions for that run.
For one trial:

```text
X = initial robot-to-ball distance
Y = final robot-to-ball distance
F = round((X - Y) / X, 2)
fitness = max(0.0, F)
```

The score is clamped to the range `[0.0, 1.0]`. If the robot reaches the ball,
the simulation stops early using the configured reached-ball threshold. The
fitness for a genotype is the average normalized score across all training ball
positions.

## Evolution

The EA is generational. For each next-generation slot, reproduction follows the
supervisor-specified tournament-clone procedure:

- The current population is re-evaluated each generation.
- Randomly select `TOURNAMENT_SIZE` genotypes from the current population.
- Select the genotype with the highest fitness as the tournament winner.
- Duplicate the winner into two clone genotypes.
- Apply one-point crossover to the two clones, producing two child genotypes.
- Randomly keep one of the two children.
- Mutate each gene in the kept child independently with
  `MUTATION_PROBABILITY`; selected genes receive Gaussian noise with standard
  deviation `MUTATE_STD`.
- Repeat until the next generation has `POPULATION_SIZE` genotypes.
- The next generation is entirely the newly produced offspring population.
- The best-ever individual is tracked for saving/logging, but is not inserted
  into the next generation as an elite unless it is recreated by reproduction.

## Outputs

Each run writes artifacts to `output/<run_name>/`:

```text
output/<run_name>/
├── gen1.pkl
├── gen2.pkl
├── ...
├── parameters_<body>.npy
├── parameters_<body>_run_<timestamp>.csv
└── generations.csv
```

`gen<N>.pkl`  
Generation snapshot. It stores the current generation's best controller, full
population, population fitnesses, fixed training ball poses, RNG states, config
source, code paths, and enough state to resume training.

`parameters_<body>.npy`  
Best-ever controller weights saved during the run.

`parameters_<body>_run_<timestamp>.csv`  
Live training CSV written during the run.

`generations.csv`  
Optional CSV exported by `run.py -o` from all `gen*.pkl` files in the output
folder.

## HPC Jobs

Cluster job scripts live under `jobs/`. They load the Python environment, set
MuJoCo/OpenGL and thread-related environment variables, and launch `run.py` with
a selected config, this experiment folder, and an output directory. Standard
output and errors are written to `jobs/logs/`.
