# Evolving Flocks Simulation

## Introduction
This repository hosts the Python code for a simulation of evolving artificial life forms, called 'boids', which display complex collective behaviors. These behaviors arise from interactions between boids and predators, driven by neural networks and evolutionary mechanisms.

## Author
**Nam H. Le, PhD.**  
Former Senior Research Fellow at the University of Southampton, currently a Postdoctoral Research Associate at the University of Vermont.  
This code is part of the project "Connectionist approaches to evolutionary transitions in individuality," conducted at the University of Southampton. The project is led by Principal Investigators Prof. Richard A. Watson (University of Southampton) and Prof. Mike Levin (Tufts University).

## Video Demonstration
Watch the demonstration of the simulation on YouTube:  
[![Watch the video](https://img.youtube.com/vi/0fqAUwj0fDk/hqdefault.jpg)](https://www.youtube.com/watch?v=0fqAUwj0fDk)


## Features
- **Boid Behaviors**: Boids follow simple rules of separation, alignment, and cohesion, which are influenced by neural networks.
- **Predator Dynamics**: Predators influence boid behaviors, prompting the emergence of defensive strategies.
- **Neural Decision Making**: Actions of each boid are governed by a neural network that evolves over time.
- **Evolutionary Pressures**: Boids reproduce based on accumulated energy, with genetic variations impacting their offspring.
- **Simulation Visualization**: Utilizes Pygame for dynamic visualization, offering real-time insights into complex interactions.

## Getting Started

### Prerequisites
- Python 3.8 or newer
- Pygame
- NumPy
- Matplotlib (for generating simulation metrics plots)

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/namlehai90/evolving-flocks-simulation.git
   ```
2. **Install required packages**:
   ```bash
   pip install pygame numpy matplotlib
   ```

### Running the Simulation
Execute the main script by navigating to the repository directory:
```bash
python evolving_flocks.py
```

## Simulation Details
- **Roles**: Boids can assume roles of 'workers' or 'soldiers', each affecting their behavior and interaction.
- **Predator Behavior**: Predators chase boids but also react to defensive formations and group sizes.
- **Energy Dynamics**: Boids' survival and reproductive success are tied to their energy management.
- **Metrics Tracking**: The simulation monitors metrics like flock numbers, average flock size, and reproduction rates.

## Modifying the Simulation
Experiment with different behaviors and evolutionary strategies by modifying:
- Number of boids or predators.
- Energy thresholds for reproduction.
- Structure of the neural networks guiding boid decisions.

## Contribution
Contributions are encouraged! If you have ideas for enhancements or find a bug, please open an issue or submit a pull request.

## License
This project is released under the MIT License.

## Acknowledgements
- Inspired by concepts from artificial life and complex system studies.
- Gratitude to all who have contributed time and effort toward enhancing this simulation.

