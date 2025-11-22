import sys
import os

# Add CARLA agents path to sys.path for imports
# Use environment variable or fallback to relative path
CARLA_AGENTS = os.environ.get('CARLA_AGENTS_PATH', r"F:\CARLA_0.9.16\PythonAPI\carla")
if os.path.exists(CARLA_AGENTS) and CARLA_AGENTS not in sys.path:
    sys.path.append(CARLA_AGENTS)



