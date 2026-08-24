# TennisSimulation4D

A discrete, physics-based tennis simulation and evolutionary search system used to explore tennis shot selection strategies through large-scale stochastic self-play.

## Overview

TennisSimulation4D is an experimental platform for investigating how shot selection strategies emerge through competition, exploration, and evolution.

The system models tennis points using:

- Discrete court positions
- Physics-based ball trajectories
- Bounce-point targeting
- Spin variation
- Defensive positioning
- Stochastic shot generation
- Evolution across generations of play

The system builds knowledge through repeated simulation, measurement, and selection.

## Core Concepts

### Discrete Search Space

All trajectory characteristics are discrete.

Examples include:

- Intercept locations
- Bounce locations
- Apex heights
- Spin values
- Defensive positions

This creates a large but finite combinatorial search space.

### Stochastic Exploration

The system continuously explores new trajectory combinations through recombination of existing trajectories.

Exploration is considered a first-class design goal.

### Evolutionary Selection

Trajectories accumulate:

- Wins
- Losses
- Counts

over multiple generations.

Successful combinations tend to survive while unsuccessful combinations gradually disappear.

### Zero-Sum Competition

Both players draw from the same underlying trajectory knowledge.

As a result:

- Average win rates remain near 50%
- Improvement comes from better decision making
- The system searches for competitive equilibria rather than absolute optimal solutions

## Project Components

### Trajectory4D

Core simulation engine.

Responsibilities include:

- Point simulation
- Shot generation
- Trajectory evaluation
- Result generation
- Evolutionary self-play

### IterativeSimulation

Simulations are generation based, and each new generation is based on the previous generation, and a file called the reference file is loaded into memory to guide the selection of trajectories to be simulated during each generation. A new reference file is created at the completion of each new generation and the reference file is smaller version that combines multiple shots with the same "context" into a single row with accumulating statistics. 

Generation size is configurable, but the default is 5 million points. A point tends to average about 5 shots which results in an approximate total of 25 million shots each of which is a row in the generation file. 

The generation files accumulate in the IterativeSimulation folder, which also contains scripts for analyzing the results. 

### TrajectoryGenerator
- Generates the library of physically accurate trajectories used by the simulation
- Library includes approximately 330,000 trajectories with the points plotting each path

### Trajectory Explorer and Solver (separate MAUI project)

Uses the simulation output data and enables the user to make four chess-like decisions and compare their selections to optimal selections

- Intercept Position Selection
- Shot Trajectory Selection
- Defensive Position Selection

Application provides feedback on how close their decisions compared to the estimated probabilities
Player can play against themselves, play against the machine, or potentially, play against another connected player.

## Current Status

This project is an active research and experimentation platform.

Many design decisions are still being evaluated and refined.

Contributors should expect:

- Rapid iteration
- Experimental algorithms
- Ongoing architectural changes

## Contributing

Contributions are welcome.

Before proposing major changes, please review the project's core assumptions:

1. The trajectory space is discrete and finite.
2. Exploration is a primary objective.
3. Stochastic recombination is intentionally used.
4. The system is zero-sum.
5. Diversity is generally preferred over early convergence.
6. Selection pressure occurs across generations.

Please open an Issue before submitting large architectural changes.

## License

This project is licensed under the MIT License.

## Author

Michael Yeager  
Blueberry Robotics LLC
