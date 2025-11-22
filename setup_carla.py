# setup_carla.py
import subprocess
import sys
import os

def install_carla_wheel():
    """Install CARLA wheel if not already installed"""
    try:
        import carla
        print("CARLA already installed")
    except ImportError:
        wheel_path = r"C:\Users\Guni\Documents\Uni\lane_change_final_proj\carla-0.9.16-cp312-cp312-win_amd64.whl"
        
        if os.path.exists(wheel_path):
            print("Installing CARLA wheel...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_path])
            print("CARLA installed")
        else:
            print(f"ERROR: Wheel not found at {wheel_path}")
            sys.exit(1)

if __name__ == "__main__":
    install_carla_wheel()
    print("\nCARLA wheel installation complete!")