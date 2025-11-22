# Smart Adaptive Cruise Control with RL-Guided Lane Changes

A multi-machine autonomous driving system featuring intelligent adaptive cruise control with reinforcement learning-based lane change decisions, built on the CARLA simulator.

## Overview

This project implements a **Smart Adaptive Cruise Control (ACC)** system that autonomously maintains target speed and performs intelligent lane changes using deep reinforcement learning. The system features a complete perception-decision-control pipeline with multi-machine coordination capabilities.

### What is Smart ACC with RL-Guided Lane Changes?

Unlike traditional cruise control that only maintains speed, this system:
- **Maintains target speed** while adapting to traffic conditions
- **Autonomously decides when to change lanes** using a trained DQN agent
- **Perceives the environment** through camera, LiDAR, and radar sensors
- **Coordinates multiple vehicles** in a shared simulation environment

The RL agent learns optimal lane change policies considering:
- Traffic density and vehicle positions
- Current lane and target lane conditions
- Speed maintenance objectives
- Safety constraints

### Multi-Machine Architecture

This system supports **distributed deployment across multiple machines**:

```
┌──────────────────────────────────────────────────────────────┐
│                    CARLA Simulator                           │
│                  (Can run on Machine A)                      │
└──────────────────┬───────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼────────┐   ┌────────▼────────┐
│ Server Process │   │  Client 1       │
│ (Machine A)    │◄─►│  (Machine B)    │
│ - ZeroMQ Coord │   │  - Perception   │
│ - Traffic Mgmt │   │  - RL Decision  │
│ - World Tick   │   │  - Control      │
└────────────────┘   └─────────────────┘
        │
        │
   ┌────▼────────┐
   │  Client 2   │
   │ (Machine C) │
   │ - Perception│
   │ - RL Decision│
   │ - Control   │
   └─────────────┘
```

**Key architectural features:**
- **One server per CARLA instance**: Manages world state, traffic, and coordination
- **N distributed clients**: Each can run on different machines with their own ego vehicle
- **ZeroMQ-based communication**: REQ-REP for state queries, PUB-SUB for event broadcasting
- **Hybrid physics**: Efficient traffic simulation with selective physics activation around each ego vehicle

## Key Features

### Decision Layer
- **DQN-based RL Agent**: Deep Q-Network for autonomous lane change decisions
- **Inference & Training Modes**: Pre-trained model inference or custom training
- **State Representation**: Comprehensive observation space including nearby vehicles, lanes, and speeds

### Advanced Perception Pipeline
- **YOLO Object Detection**: Vehicle and pedestrian detection using YOLO11 models
- **Traffic Light Classification**: Dedicated YOLO classifier for traffic light state recognition
- **Lane Detection**: Ultra-Fast Lane Detection v2 (UFLDv2) with 40% performance optimization via CUDA and vectorization
- **Multi-Sensor Fusion**: Camera, LiDAR, and Radar integration
- **Sensor Failover System**: Automatic sensor switching for robust operation

### Control & Coordination
- **Adaptive Cruise Control**: Speed maintenance with traffic awareness
- **Real-time Control**: Keyboard controls for target speed adjustment
- **Multi-Client Coordination**: ZeroMQ-based synchronization
- **KPI Tracking**: Collision monitoring and performance metrics

## Installation

### Prerequisites

- **Python 3.12**
- **CARLA Simulator 0.9.16**
- **CUDA-capable GPU** (recommended for real-time performance)
- **Windows 64-bit** (current configuration)

### Step 1: CARLA Setup

1. Download CARLA 0.9.16 from [CARLA Releases](https://github.com/carla-simulator/carla/releases)
2. Extract to your preferred location (e.g., `C:\CARLA_0.9.16`)
3. Note the installation path for later configuration

### Step 2: Directory Structure Setup

Your project directory should be organized as follows:

```
parent_folder/
├── lane_change/                          # This repository
│   ├── src/
│   ├── config/
│   ├── demo.py
│   └── ...
├── Ultra-Fast-Lane-Detection-v2/         # UFLDv2 repository (clone separately)
│   ├── model/
│   ├── utils/
│   └── ...
└── ufld_weights/                         # UFLDv2 model weights
    └── culane_res34.pth
```

### Step 3: Clone UFLDv2 Repository

In the **parent folder** of this project:

```bash
cd ..  # Go to parent directory
git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2.git
```

### Step 4: Download UFLDv2 Weights

1. Download the CULane ResNet34 weights from [UFLDv2 releases](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/releases)
2. Create `ufld_weights/` directory in the parent folder
3. Place `culane_res34.pth` in `ufld_weights/`

```bash
mkdir ufld_weights
# Download culane_res34.pth and place it in ufld_weights/
```

### Step 5: Python Environment Setup

1. Create a conda environment:
```bash
conda create -n lane_change python=3.12
conda activate lane_change
```

2. Install PyTorch with CUDA support:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. Clone and install this package:
```bash
git clone <repository-url> lane_change
cd lane_change
pip install -e .
```

4. Install CARLA Python API:

Edit `setup_carla.py` to point to your CARLA wheel location:
```python
wheel_path = r"C:\path\to\your\carla-0.9.16-cp312-cp312-win_amd64.whl"
```

Then run:
```bash
python setup_carla.py
```

### Dependencies

Core dependencies (automatically installed via setup.py):
- **ultralytics**: YOLO detection and classification models
- **torch, torchvision**: Deep learning framework for DQN and YOLO
- **opencv-python**: Image processing
- **numpy**: Numerical computing
- **shapely**: Geometric operations for lane/vehicle calculations
- **pyzmq**: Multi-machine communication
- **pygame**: Visualization and keyboard controls
- **pyyaml**: Configuration management
- **psutil**: Process priority management
- **scipy, matplotlib, pandas, scikit-learn**: Scientific computing and analysis

## Configuration

The system is configured via `config/carla_config.yaml`:

### CARLA Connection
```yaml
carla:
  host: "localhost"        # Change for multi-machine setup
  port: 2000
  timeout: 120.0
  map_name: "Town12"       # Highway map for lane changes
```

### Vehicle Spawn
```yaml
ego_vehicle:
  blueprint: "vehicle.tesla.model3"
  spawn_location:
    x: -385.80
    y: 3100.61
```

### Sensors Configuration
```yaml
sensors:
  camera:
    image_size_x: 640
    image_size_y: 360
    fov: 120

  failover:
    enabled: true
    timeout_frames: 5
    consecutive_failures: 5
```

### Traffic Settings
```yaml
traffic:
  num_vehicles: 20                        # NPC vehicles
  global_distance_to_leading_vehicle: 2.5
  global_speed_difference: 0.0            # Speed variation
```

### Multi-Client Server
```yaml
server:
  num_initial_clients: 1                  # Number of clients to wait for
  registration_timeout: 60
  zeromq_rep_port: 5555                   # State query port
  zeromq_pub_port: 5556                   # Event broadcast port

  hybrid_physics:
    enabled: true
    radius: 70.0                          # Physics activation radius per vehicle

client:
  server_host: "localhost"                # Change to server IP for multi-machine
  server_rep_port: 5555
  server_pub_port: 5556
```

### Perception & Decision
Configuration in `src/lane_change/config/`:
- `perception_config.py`: YOLO models, lane detection parameters
- `decision_config.py`: DQN hyperparameters, training settings

## Models

The system uses three types of models:

### 1. Object Detection (YOLO)
Pre-trained YOLO models for vehicle and pedestrian detection:

Place in project root or `models/` directory.

### 2. Traffic Light Classification (YOLO)
Specialized YOLO classifier for traffic light state detection:

Trained on traffic light states: Red, Yellow, Green

### 3. Lane Detection (UFLDv2)
Ultra-Fast Lane Detection v2 with CUDA optimizations:
- `culane_res34.pth`: ResNet34 backbone trained on CULane dataset
- Located in `../ufld_weights/culane_res34.pth`

### 4. RL Agent (DQN)
Deep Q-Network for lane change decisions:
- Trained via `train.py`
- Saved checkpoints in project directory
- Configurable in `src/lane_change/config/decision_config.py`

**Note:** For training your own models, update paths in configuration files:
- CARLA installation path (for data collection)
- Traffic light dataset path (if training traffic light classifier)

## Usage

### Single Machine Demo

The simplest way to run the system on one machine:

1. **Start CARLA Simulator:**
```bash
cd C:\CARLA_0.9.16
CarlaUE4.exe -quality-level=Low
```

2. **Run Demo (standalone mode):**
```bash
python demo.py
```

The system will:
- Connect to CARLA
- Spawn ego vehicle and traffic
- Run perception pipeline
- Use RL agent to make lane change decisions
- Maintain adaptive cruise control
- User can press up and down to change the target speed that the car will maintain

### Multi-Client Mode (Single or Multi-Machine)

For coordinated multi-vehicle scenarios:

1. **Start CARLA Simulator** (on Machine A):
```bash
cd C:\CARLA_0.9.16
CarlaUE4.exe -quality-level=Low
```

2. **Start Server** (on Machine A):
```bash
python carla_server.py --num-clients 2 --map Town12
```

Server will:
- Load the specified map
- Configure hybrid physics
- Spawn global traffic
- Wait for clients to connect
- Coordinate simulation ticks at 20 Hz

3. **Start Clients** (on any machine):

On Machine B:
```bash
# Edit config/carla_config.yaml:
# carla.host: "<Machine_A_IP>"
# client.server_host: "<Machine_A_IP>"

python demo.py
```

On Machine C:
```bash
# Same config changes
python demo.py
```

**Multi-Machine Setup:**
- Ensure all machines can reach CARLA server (port 2000)
- Ensure clients can reach server (ports 5555, 5556)
- Update `carla.host` and `client.server_host` in config

**Synchronization:**
- Server waits for `num_initial_clients` to connect
- Sends GO signal when all clients ready
- Late joiners can connect after initial sync

### Training Mode

Train your own DQN agent:

1. **Configure Training** in `src/lane_change/config/decision_config.py`:
```python
DecisionConfig(
    learning_rate=1e-4,
    gamma=0.99,
    batch_size=64,
    buffer_size=100000,
    # ... other hyperparameters
)
```

2. **Start Training Server:**
```bash
python training_server.py
```

This spawns the CARLA environment for training episodes.

3. **Run Training:**
```bash
python train.py
```

Training will:
- Collect experiences via interaction with CARLA
- Train DQN network
- Save checkpoints periodically
- Log to TensorBoard

**Note:** For custom traffic light classifier training, prepare your dataset and update paths in perception config.

## Project Structure

```
lane_change/
├── config/
│   └── carla_config.yaml              # Main configuration
├── src/lane_change/
│   ├── config/
│   │   ├── perception_config.py       # Perception settings
│   │   └── decision_config.py         # RL settings
│   ├── perception/                    # Perception pipeline
│   │   ├── yolo_detector.py           # YOLO object detection
│   │   ├── lane_detector.py           # UFLDv2 integration
│   │   ├── traffic_light_yolo_classifier.py  # Traffic light YOLO
│   │   ├── perception_core.py         # Pipeline coordinator
│   │   └── ...
│   ├── decision/                      # RL decision layer
│   │   ├── dqn_network.py             # DQN architecture
│   │   ├── rl_agent.py                # Training & inference agents
│   │   ├── rl_trainer.py              # Training loop
│   │   ├── decision_core.py           # Decision logic
│   │   └── ...
│   ├── gateway/                       # Abstraction layer
│   │   ├── detector.py                # Detector interfaces
│   │   ├── sensor_manager.py          # Sensor coordination
│   │   ├── vehicle_controller.py      # Control interface
│   │   ├── traffic_manager.py         # Traffic interface
│   │   ├── server_client.py           # ZeroMQ client
│   │   └── ...
│   └── plant/                         # CARLA interface
│       ├── carla_base_server.py       # Server base
│       ├── carla_vehicle_manager.py   # Vehicle management
│       ├── carla_sensor_manager.py    # Sensor management
│       ├── carla_traffic_manager.py   # Traffic management
│       └── ...
├── models/                            # Pre-trained models (optional)
├── scripts/                           # Utility scripts
│   └── traffic_light_classifier/      # Traffic light training
├── demo.py                            # Main demo client
├── carla_server.py                    # Multi-client server
├── train.py                           # RL training script
├── training_server.py                 # Training environment
├── setup.py                           # Package installation
└── README.md                          # This file
```

## Key Performance Indicators (KPIs)

The system tracks performance metrics:

1. **Collision Rate**: Number of collisions per episode
2. **Speed Maintenance**: Deviation from target speed
3. **Lane Change Success Rate**: Percentage of successful lane changes
4. **Sensor Health**: Failover events and data quality metrics

View KPIs in the console output during execution.

## Performance Optimizations

- **UFLDv2 Lane Detection**: 40% performance improvement through CUDA optimizations and vectorization
- **Hybrid Physics**: Selective physics activation (70m radius) reduces computational load
- **Multi-threaded Perception**: Parallel processing of detection tasks
- **Process Priority**: High priority for real-time performance (`psutil.HIGH_PRIORITY_CLASS`)

## Development

### Code Standards

See `CLAUDE.md` for development guidelines:
- Follow SOLID principles and Single Source of Truth (SSOT)
- Write clean, production-style code focused on happy paths
- Prioritize readable and maintainable code
- Use GPU optimizations when beneficial (analyze overhead vs. gain)
- No backward compatibility requirements

### Testing

Run tests:
```bash
pytest tests/
```

## Troubleshooting

### CARLA Connection Issues
- Ensure CARLA is running before starting server/clients
- Check firewall settings for ports 2000, 5555, 5556
- Verify map name matches available CARLA maps
- For multi-machine: ensure network connectivity

### UFLDv2 Import Errors
- Verify `Ultra-Fast-Lane-Detection-v2` is cloned in parent folder
- Check weights path: `../ufld_weights/culane_res34.pth`
- Ensure UFLDv2 directory structure is intact

### GPU/CUDA Issues
- Verify CUDA 12.1 installation
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce batch sizes if out of GPU memory
- Lower number of traffic vehicles in config

### Sensor Failover Activation
- Check sensor configurations in `carla_config.yaml`
- Review logs for sensor validation failures
- Adjust `timeout_frames` and `consecutive_failures` parameters
- Ensure sensors are properly attached to vehicle

### Multi-Client Issues
- Verify all clients use same CARLA server IP
- Check ZeroMQ port availability (5555, 5556)
- Ensure server waits for correct number of clients
- Check network latency for multi-machine setups

## Contributing

1. Follow code standards in `CLAUDE.md`
2. Ensure all tests pass
3. Update documentation for new features
4. Use descriptive commit messages

## License

[Add your license here]

## Acknowledgments

- **CARLA Simulator Team**: Open-source autonomous driving simulator
- **Ultralytics**: YOLO object detection and classification models
- **CFZd**: Ultra-Fast Lane Detection v2 implementation
- **PyTorch**: Deep learning framework

## Contact

[Add contact information]
