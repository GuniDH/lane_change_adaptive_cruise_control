"""
CARLA Server - Multi-client open-world server with ZeroMQ coordination.

This is the central authority for open-world mode:
- Loading maps
- Configuring world settings and hybrid physics
- Spawning global traffic
- Ticking simulation at 20 Hz
- Coordinating multiple clients via ZeroMQ

Architecture:
  ONE server per CARLA instance
  N ego clients connect independently
  Server coordinates synchronized start with GO signal
  Late joiners supported after initial sync

Usage:
    python carla_server.py --num-clients 2 --map Town12
"""

import sys
import zmq
import carla
import time
import yaml
import logging
import argparse
import signal
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lane_change.plant.carla_base_server import CarlaBaseServer
from lane_change.plant.carla_traffic_manager import CarlaTrafficManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CarlaServer(CarlaBaseServer):
    """
    Multi-client open-world server with ZeroMQ coordination.

    Responsibilities:
    - Connect to CARLA and setup world
    - Configure hybrid physics for traffic management
    - Spawn global traffic vehicles
    - Start ZeroMQ servers (REQ-REP + PUB-SUB)
    - Wait for N clients (60s timeout)
    - Broadcast GO signal
    - Tick continuously at 20 Hz
    - Handle state queries from late joiners
    """

    def __init__(self, host='localhost', port=2000, config=None, num_clients=1):
        """
        Initialize CARLA server.

        Args:
            host: CARLA server host
            port: CARLA server port
            config: Configuration dictionary
            num_clients: Number of initial clients to wait for
        """
        super().__init__(host, port, config)
        self.num_clients = num_clients

        # Server state
        self.phase = "waiting"  # "waiting" or "running"
        self.ready_clients = set()  # Set of client_ids that have signaled ready
        self.connected_clients = set()  # Set of all connected client_ids
        self.next_client_id = 0  # Auto-increment counter for ID assignment
        self.traffic_manager = None  # Created after connecting to CARLA
        self.shutdown_in_progress = False
        self.cleanup_in_progress = False  # Pause ticking during cleanup

        # ZeroMQ sockets
        self.zmq_context = None
        self.rep_socket = None  # REQ-REP for state queries
        self.pub_socket = None  # PUB-SUB for event broadcasting

    def setup_zeromq(self):
        """Start ZeroMQ servers for client communication."""
        server_config = self.config.get('server', {})
        rep_port = server_config.get('zeromq_rep_port', 5555)
        pub_port = server_config.get('zeromq_pub_port', 5556)

        # Create ZeroMQ context
        self.zmq_context = zmq.Context()

        # REP socket for state queries (request-reply pattern)
        self.rep_socket = self.zmq_context.socket(zmq.REP)
        self.rep_socket.bind(f"tcp://*:{rep_port}")
        self.rep_socket.setsockopt(zmq.RCVTIMEO, 0)  # Non-blocking (immediate return)
        logger.info(f"ZeroMQ REP server listening on port {rep_port}")

        # PUB socket for event broadcasting (publish-subscribe pattern)
        self.pub_socket = self.zmq_context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://*:{pub_port}")
        logger.info(f"ZeroMQ PUB server listening on port {pub_port}")

    def setup_hybrid_physics(self):
        """
        Configure Traffic Manager with hybrid physics mode.

        CRITICAL: Must be called BEFORE spawning any ego vehicles!
        """
        server_config = self.config.get('server', {})
        hybrid_config = server_config.get('hybrid_physics', {})

        if not hybrid_config.get('enabled', True):
            logger.info("Hybrid physics disabled")
            return

        # Enable hybrid physics mode
        self.carla_tm.set_hybrid_physics_mode(True)

        # Set physics radius around each ego vehicle
        radius = hybrid_config.get('radius', 70.0)
        self.carla_tm.set_hybrid_physics_radius(radius)

        logger.info(f"Hybrid physics enabled: radius={radius}m")

    def handle_client_messages(self):
        """
        Handle incoming messages from clients (non-blocking).

        Processes registration, state queries, and ready signals.
        """
        try:
            message = self.rep_socket.recv_json()

            if message.get("type") == "register":
                # Client requesting registration and ID assignment
                assigned_id = self.next_client_id
                self.next_client_id += 1

                # Track connected client
                self.connected_clients.add(assigned_id)

                logger.info(f"Registered new client with client_id {assigned_id} (total connected: {len(self.connected_clients)})")

                response = {
                    "assigned_client_id": assigned_id,
                    "phase": self.phase
                }
                try:
                    self.rep_socket.send_json(response)
                except zmq.ZMQError as e:
                    logger.warning(f"Failed to send response (client disconnected?): {e}")

            elif message.get("type") == "get_state":
                # Client querying current state (read-only, no side effects)
                response = {
                    "phase": self.phase,
                    "carla_host": self.host,
                    "carla_port": self.port
                }
                try:
                    self.rep_socket.send_json(response)
                except zmq.ZMQError as e:
                    logger.warning(f"Failed to send response (client disconnected?): {e}")

            elif message.get("type") == "ready":
                # Client signaling ready to start
                client_id = message.get("client_id")

                if client_id is not None:
                    self.ready_clients.add(client_id)
                    logger.info(f"Client {client_id} signaled ready ({len(self.ready_clients)}/{self.num_clients})")

                try:
                    self.rep_socket.send_json({
                        "status": "acknowledged"
                    })
                except zmq.ZMQError as e:
                    logger.warning(f"Failed to send response (client disconnected?): {e}")

            elif message.get("type") == "cleanup_request":
                # Client requesting server to clean up its actors
                client_id = message.get("client_id")
                ego_actor_id = message.get("ego_actor_id")

                logger.info(f"Client {client_id} requesting cleanup (ego_actor_id={ego_actor_id})")

                # Pause ticking during cleanup to prevent streaming race
                self.cleanup_in_progress = True

                # Destroy the client's ego vehicle and sensors
                if ego_actor_id is not None:
                    try:
                        ego_actor = self.world.get_actor(ego_actor_id)
                        if ego_actor and ego_actor.is_alive:
                            # Find and destroy child sensors
                            sensors = [child for child in self.world.get_actors()
                                     if child.parent and child.parent.id == ego_actor_id]

                            for sensor in sensors:
                                try:
                                    sensor.destroy()
                                except Exception as e:
                                    logger.warning(f"Error destroying sensor {sensor.id}: {e}")

                            # Destroy vehicle
                            ego_actor.destroy()
                            logger.info(f"Destroyed client {client_id}'s actors (vehicle + {len(sensors)} sensors)")
                        else:
                            logger.warning(f"Ego actor {ego_actor_id} not found")
                    except Exception as e:
                        logger.error(f"Error during cleanup: {e}", exc_info=True)

                # Resume ticking
                self.cleanup_in_progress = False

                try:
                    self.rep_socket.send_json({"status": "cleanup_complete"})
                except zmq.ZMQError as e:
                    logger.warning(f"Failed to send cleanup response: {e}")

            elif message.get("type") == "disconnect":
                # Client signaling clean disconnect
                client_id = message.get("client_id")

                if client_id is not None and client_id in self.connected_clients:
                    self.connected_clients.remove(client_id)
                    logger.info(f"Client {client_id} disconnected ({len(self.connected_clients)} remaining)")

                    # Shutdown server when last client disconnects
                    if len(self.connected_clients) == 0:
                        logger.info("Last client disconnected - initiating graceful shutdown")
                        self.running = False

                try:
                    self.rep_socket.send_json({"status": "goodbye"})
                except zmq.ZMQError as e:
                    logger.warning(f"Failed to send goodbye: {e}")

        except zmq.Again:
            # No message received (timeout) - normal during ticking
            pass
        except Exception as e:
            logger.error(f"Error handling client message: {e}")

    def broadcast_shutdown(self):
        """Broadcast shutdown signal to all clients."""
        logger.info("Broadcasting SHUTDOWN signal to all clients...")

        shutdown_event = {
            "event": "SHUTDOWN",
            "message": "Server shutting down - please disconnect",
            "timeout": 10
        }
        self.pub_socket.send_json(shutdown_event)
        time.sleep(0.5)  # Give clients time to receive

    def wait_for_clients_disconnect(self, timeout=15):
        """Wait for all clients to disconnect gracefully."""
        if len(self.connected_clients) == 0:
            logger.info("No clients connected - skipping disconnect wait")
            return

        logger.info(f"Waiting for {len(self.connected_clients)} client(s) to disconnect...")

        start_time = time.time()

        while len(self.connected_clients) > 0:
            elapsed = time.time() - start_time

            if elapsed > timeout:
                logger.warning(f"Timeout! {len(self.connected_clients)} client(s) didn't disconnect gracefully")
                break

            # Keep handling disconnect messages
            self.handle_client_messages()

            # Keep ticking world so clients can cleanup
            try:
                self.tick_world()
            except Exception as e:
                logger.warning(f"Error ticking during shutdown: {e}")
                break

            time.sleep(0.05)

        if len(self.connected_clients) == 0:
            logger.info("✓ All clients disconnected gracefully")
        else:
            logger.warning(f"✗ {len(self.connected_clients)} client(s) still connected: {self.connected_clients}")

    def wait_for_clients_and_broadcast_go(self, timeout=60):
        """
        Wait for N clients to signal ready OR timeout.

        After GO:
        - Broadcast GO event via PUB socket
        - Enable traffic autopilot
        - Change phase to "running"

        Args:
            timeout: Maximum seconds to wait (default: 60)
        """
        server_config = self.config.get('server', {})
        timeout = server_config.get('registration_timeout', timeout)

        logger.info("="*60)
        logger.info(f"WAITING PHASE: Waiting for {self.num_clients} client(s)")
        logger.info(f"Timeout: {timeout} seconds")
        logger.info("="*60)

        start = time.time()

        while time.time() - start < timeout:
            # Handle client messages
            self.handle_client_messages()

            # Check if all clients ready
            if len(self.ready_clients) >= self.num_clients:
                logger.info(f"All {self.num_clients} client(s) ready!")
                break

            # Tick world while waiting (clients need ticks to spawn)
            self.tick_world()
            time.sleep(0.01)  # Small delay to prevent busy loop

        # All clients ready - ego vehicles spawned
        # NOW spawn traffic at the real hero location
        server_config = self.config.get('server', {})
        traffic_config = server_config.get('global_traffic', {})
        num_vehicles = traffic_config.get('num_vehicles', 100)

        if num_vehicles > 0:
            # Get real hero location from spawned actors
            hero_location = None
            for actor in self.world.get_actors().filter('vehicle.*'):
                if actor.attributes.get('role_name') == 'hero':
                    hero_location = actor.get_location()
                    logger.info(f"Found hero at ({hero_location.x:.1f}, {hero_location.y:.1f})")
                    break

            if hero_location:
                try:
                    # Spawn traffic at actual hero location
                    self.traffic_manager.spawn_traffic(
                        reference_location=hero_location,
                        num_vehicles=num_vehicles,
                        distance_range=(-50.0, 50.0),
                        speed_range=(20.0, 80.0),
                        min_spacing=0.0,
                        configs=[
                            (-3.0, 40.0, 'left'),
                            (3.0, 40.0, 'left'),
                            (-3.0, 40.0, 'right'),
                            (3.0, 40.0, 'right'),
                            (60.0, 20.0, 'ego'),
                        ]
                    )
                    logger.info(f"Spawned {self.traffic_manager.get_vehicle_count()} traffic vehicles at hero location")

                    # Configure autopilot
                    speed_variation = traffic_config.get('autopilot_speed_variation', 30.0)
                    self.carla_tm.global_percentage_speed_difference(speed_variation)
                except Exception as e:
                    logger.error(f"Error spawning traffic: {e}")
            else:
                logger.warning("No hero vehicle found - cannot spawn traffic")

        # Broadcast GO signal
        logger.info("="*60)
        logger.info(f"Broadcasting GO signal ({len(self.ready_clients)}/{self.num_clients} clients ready)")
        logger.info("="*60)

        go_event = {
            "event": "GO",
            "phase": "running"
        }
        self.pub_socket.send_json(go_event)

        # Apply traffic speeds and enable autopilot (synchronized)
        if self.traffic_manager and self.traffic_manager.get_vehicle_count() > 0:
            logger.info("Initializing traffic vehicles...")
            self.traffic_manager.apply_speeds()
            self.traffic_manager.enable_autopilot()
        else:
            logger.info("No traffic vehicles to initialize")

        # Change phase to running
        self.phase = "running"
        logger.info("Phase changed to RUNNING - simulation active")

    def run(self):
        """
        Main server loop.

        Phase 1: Wait for clients + broadcast GO
        Phase 2: Continuous ticking + handle late joiner queries
        """
        logger.info("="*60)
        logger.info("CARLA Server - Starting Open-World Mode")
        logger.info("="*60)

        # Wait for initial clients and broadcast GO
        self.wait_for_clients_and_broadcast_go()

        # Main simulation loop
        self.running = True
        fps_start = time.time()
        fps_count = 0
        frame_count = 0
        carla_connection_failures = 0
        max_connection_failures = 3

        logger.info("="*60)
        logger.info("RUNNING PHASE: Simulation active")
        logger.info("="*60)

        try:
            while self.running:
                try:
                    # Handle messages first (may set cleanup_in_progress flag)
                    self.handle_client_messages()

                    # Only tick if not in cleanup (prevents streaming race)
                    if not self.cleanup_in_progress:
                        # Tick world at 20 Hz
                        self.tick_world()
                        frame_count += 1
                        fps_count += 1

                        # Reset failure counter on successful tick
                        carla_connection_failures = 0
                    else:
                        # During cleanup, sleep briefly to prevent busy loop
                        time.sleep(0.01)

                    # Log FPS every 5 seconds
                    elapsed = time.time() - fps_start
                    if elapsed >= 5.0:
                        fps = fps_count / elapsed
                        logger.info(f"Simulation FPS: {fps:.1f} | Frame: {frame_count}")
                        fps_start = time.time()
                        fps_count = 0

                except RuntimeError as e:
                    # CARLA connection lost (e.g., CarlaUE4.exe closed)
                    carla_connection_failures += 1
                    logger.error(f"CARLA connection error ({carla_connection_failures}/{max_connection_failures}): {e}")

                    if carla_connection_failures >= max_connection_failures:
                        logger.error("="*60)
                        logger.error("CARLA CONNECTION LOST")
                        logger.error(f"Failed {max_connection_failures} consecutive ticks")
                        logger.error("CARLA executable (CarlaUE4.exe) may have been closed")
                        logger.error("Shutting down server...")
                        logger.error("="*60)
                        self.running = False
                        break

                    # Brief delay before retry
                    time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Server interrupted by user (Ctrl+C)")

        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)

        finally:
            # Always perform graceful shutdown sequence on ANY exit path
            logger.info("=" * 60)
            logger.info("GRACEFUL SHUTDOWN SEQUENCE")
            logger.info("=" * 60)

            if len(self.connected_clients) > 0:
                self.broadcast_shutdown()
                self.wait_for_clients_disconnect(timeout=15)
            else:
                logger.info("No clients connected - skipping graceful shutdown")

    def cleanup(self):
        """Clean up ZeroMQ sockets and CARLA resources."""
        logger.info("Starting server cleanup...")

        # Destroy traffic vehicles
        if self.traffic_manager:
            try:
                self.traffic_manager.cleanup()
            except Exception as e:
                logger.error(f"Error destroying traffic vehicles: {e}")

        # Close ZeroMQ sockets
        if self.rep_socket:
            self.rep_socket.close()
        if self.pub_socket:
            self.pub_socket.close()
        if self.zmq_context:
            self.zmq_context.term()
        logger.info("ZeroMQ sockets closed")

        # Call base class cleanup (restores world settings)
        super().cleanup()


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully - let exception propagate to cleanup."""
    logger.info("Shutdown signal received - triggering KeyboardInterrupt")
    raise KeyboardInterrupt()


def main():
    parser = argparse.ArgumentParser(
        description='CARLA Server - Multi-client open-world server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Wait for 2 clients
  python carla_server.py --num-clients 2

  # Load specific map
  python carla_server.py --map Town12 --num-clients 2

  # Connect to remote CARLA
  python carla_server.py --host 192.168.1.100 --port 2000 --num-clients 3
        """
    )
    parser.add_argument('--host', default='localhost',
                       help='CARLA server host (default: localhost)')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA server port (default: 2000)')
    parser.add_argument('--map', default=None,
                       help='Map to load (default: from config)')

    # Load config to get default num_clients
    config_path = Path(__file__).parent / "config" / "carla_config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        default_num_clients = config.get('server', {}).get('num_initial_clients', 1)
    except Exception:
        default_num_clients = 2

    parser.add_argument('--num-clients', type=int, default=default_num_clients,
                       help=f'Number of initial clients to wait for (default: {default_num_clients} from config)')
    args = parser.parse_args()

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Create server
    server = CarlaServer(
        host=args.host,
        port=args.port,
        num_clients=args.num_clients
    )

    try:
        # Connect to CARLA
        server.connect_to_carla(map_name=args.map)

        # Setup Traffic Manager (must be before hybrid physics)
        server.setup_traffic_manager(synchronous_mode=True)

        # Enable synchronous mode
        server.enable_synchronous_mode(delta_seconds=0.05)

        # CRITICAL: Setup hybrid physics BEFORE spawning anything
        server.setup_hybrid_physics()

        # Create traffic controller (traffic spawned later after clients join)
        server.traffic_manager = CarlaTrafficManager(
            world=server.world,
            client=server.client,
            traffic_manager=server.carla_tm
        )
        logger.info("Traffic controller initialized (traffic will spawn after clients join)")

        # Setup ZeroMQ communication
        server.setup_zeromq()

        # Run main loop
        server.run()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        server.cleanup()
        logger.info("Server shutdown complete")


if __name__ == '__main__':
    main()
