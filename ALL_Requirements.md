# Revolve2 — Complete Requirements & Installation Guide (Linux)

## 1. System Requirements

### Operating System
- **Recommended**: Ubuntu 20.04 / 22.04 LTS (or any Debian-based Linux)
- macOS and Windows are supported but not officially assisted

### Python
- **Python 3.10.x or 3.11.x** (3.12+ is NOT supported)
- Recommended: Python 3.11

---

## 2. System-Level Dependencies

Install via apt:

```bash
sudo apt update
sudo apt install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git
```

These are required for:
- Python development headers (Cython compilation)
- C/C++ compilers (building native extensions)
- OpenGL libraries (MuJoCo rendering)
- OpenCV dependencies

### For Headless Servers (no display)
If running on a server without a monitor, install a virtual display:

```bash
sudo apt install -y xvfb
```

Then run simulations with:
```bash
xvfb-run python main.py
```

---

## 3. Python Environment Setup

```bash
# Install virtualenv
pip install virtualenv

# Create a virtual environment in the repo root
python3.11 -m virtualenv .venv

# Activate it
source .venv/bin/activate
```

> **Important**: Always activate the venv before running anything. Never use conda — it is not officially supported and causes issues.

---

## 4. Revolve2 Package Installation

From the repo root, run:

```bash
sh student_install.sh
```

This installs all 7 Revolve2 packages in editable mode plus all example dependencies.

### Manual Installation Order (if needed)

Packages must be installed bottom-up due to dependencies:

```bash
pip install -e simulation
pip install -e modular_robot
pip install -e modular_robot_simulation
pip install -e experimentation
pip install -e simulators/mujoco_simulator
pip install -e ci_group
pip install -e modular_robot_physical[remote]
```

---

## 5. Revolve2 Packages

| Package | Description | Key Requires |
|---|---|---|
| `simulation` | Physics simulation abstraction layer | — |
| `modular_robot` | Robot body/brain definitions | — |
| `modular_robot_simulation` | Scene setup and simulation | `modular_robot`, `simulation` |
| `experimentation` | EA tools and database utilities | — |
| `simulators/mujoco_simulator` | MuJoCo physics engine implementation | `simulation` |
| `ci_group` | CI Group research standards (robots, terrains, fitness) | `modular_robot_simulation` |
| `modular_robot_physical` | Physical robot hardware control | `modular_robot` |

---

## 6. Python Dependencies (auto-installed)

### Core

| Library | Version | Purpose |
|---|---|---|
| `numpy` | `^1.21.2` | Array math |
| `pyrr` | `^0.10.3` | 3D math and quaternions |
| `scipy` | `^1.7.1` | Scientific computing |
| `sqlalchemy` | `^2.0.0` | Database ORM (SQLite) |

### Simulation

| Library | Version | Purpose |
|---|---|---|
| `mujoco` | `^2.2.0` | Physics simulator |
| `dm-control` | `^1.0.3` | MuJoCo Python bindings |
| `mujoco-python-viewer` | `^0.1.3` | MuJoCo viewer |
| `opencv-python` | `^4.6.0` | Image processing and video recording |
| `opencv-contrib-python` | `^4.9.0` | Extended OpenCV modules |

### CI Group / Evolution

| Library | Version | Purpose |
|---|---|---|
| `multineat` | `^0.12` | CPPN/NEAT neuroevolution |
| `noise` | `^1.2.2` | Perlin noise for terrain generation |
| `Cython` | `^3.0.4` | C extension compilation |
| `setuptools` | `^68.2.2` | Build tools |

### Physical Robot (optional)

| Library | Version | Purpose |
|---|---|---|
| `pycapnp` | `^2.0.0b2` | Cap'n Proto serialization for robot comms |
| `typed-argparse` | `^0.3.1` | CLI argument parsing |
| `pigpio` | `^1.78` | Raspberry Pi GPIO (botv1 only) |
| `revolve2-robohat` | `0.5.0` | RoboHAT hardware (botv2 only) |

### Example-Specific

| Library | Version | Required By |
|---|---|---|
| `pandas` | `>=2.1.0` | Examples 4b, 4d, 4f (database plotting) |
| `matplotlib` | `>=3.8.0` | Examples 4b, 4d, 4f (plotting) |
| `cma` | `>=3.3.0` | Examples 4e, 4f (CMA-ES optimizer) |

---

## 7. Development / Testing Dependencies (optional)

### Testing
```bash
pip install pytest~=7.4.2 pytest-mock==3.11.1
```

### Code Quality Tools
```bash
pip install -r codetools/requirements.txt
```
Includes: `black`, `isort`, `mypy`, `pydocstyle`, `darglint`, `pyflakes`

### Documentation
```bash
pip install sphinx==7.2.6 sphinx-rtd-theme==1.3.0 sphinx-autoapi==3.0.0
```

---

## 8. Running Examples

### Basic simulation (with viewer)
```bash
python examples/1_simulator_basics/1a_simulate_single_robot/main.py
```

### Headless (no display)
```bash
xvfb-run python examples/1_simulator_basics/1a_simulate_single_robot/main.py
```

> **macOS note**: Use `mjpython` instead of `python` to launch the viewer on macOS.

---

## 9. Notes

- Do **not** use conda environments — they are not officially supported
- Always run scripts from the **repo root**, not from inside example subdirectories
- The `ci_group` package compiles a Cython extension at install time — ensure `gcc` and Python dev headers are installed before running `student_install.sh`
- MuJoCo 3.x is compatible with this codebase (the `pyproject.toml` pins `<3.0.0` but this can be overridden)
