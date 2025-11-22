"""
Training Server - Minimal CARLA server for single-client RL training.

This server provides basic CARLA world management for training:
- Connects to CARLA
- Enables synchronous mode (20 Hz)
- Ticks continuously
- NO multi-client coordination
- NO traffic spawning (client handles scenarios)
- NO ZeroMQ (no coordination needed)

Usage:
    python training_server.py --map Town12
"""

import sys
import signal
import logging
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lane_change.plant.carla_base_server import CarlaBaseServer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingServer(CarlaBaseServer):
    """
    Minimal CARLA server for single-client training.

    Responsibilities:
    - Connect to CARLA
    - Enable synchronous mode
    - Tick at 20 Hz continuously

    Does NOT:
    - Spawn traffic (client handles training scenarios)
    - Coordinate multiple clients (single-client only)
    - Use ZeroMQ (no coordination needed)
    """

    def __init__(self, host='localhost', port=2000, config=None):
        """
        Initialize training server.

        Args:
            host: CARLA server host
            port: CARLA server port
            config: Configuration dictionary
        """
        super().__init__(host, port, config)
        self.frame_count = 0

    def run(self):
        """
        Main server loop - just tick continuously.

        Client handles all scenario logic (spawn_traffic, destroy_traffic, etc.)
        """
        logger.info("="*60)
        logger.info("Training Server - Running")
        logger.info("="*60)
        logger.info("Ticking at 20 Hz - client manages all scenarios")

        self.running = True

        try:
            while self.running:
                self.tick_world()
                self.frame_count += 1

                # Log progress every 100 frames (5 seconds at 20 Hz)
                if self.frame_count % 100 == 0:
                    logger.debug(f"Frame: {self.frame_count}")

        except KeyboardInterrupt:
            logger.info("Training server interrupted by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"Training server error: {e}", exc_info=True)


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    logger.info("Shutdown signal received")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Training Server - Minimal CARLA server for single-client RL training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start training server
  python training_server.py

  # Load specific map
  python training_server.py --map Town12

  # Connect to remote CARLA
  python training_server.py --host 192.168.1.100 --port 2000
        """
    )
    parser.add_argument('--host', default='localhost',
                       help='CARLA server host (default: localhost)')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA server port (default: 2000)')
    parser.add_argument('--map', default=None,
                       help='Map to load (default: from config or current map)')
    args = parser.parse_args()

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Create and run training server
    server = TrainingServer(host=args.host, port=args.port)

    try:
        # Connect to CARLA
        server.connect_to_carla(map_name=args.map)

        # Setup Traffic Manager (but don't spawn traffic)
        server.setup_traffic_manager(synchronous_mode=True)

        # Enable synchronous mode
        server.enable_synchronous_mode(delta_seconds=0.05)

        # Run main loop
        server.run()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        server.cleanup()
        logger.info("Training server shutdown complete")


if __name__ == '__main__':
    main()
