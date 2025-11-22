"""
Server Client - ZeroMQ client for communicating with CARLA server.

This module provides ZeroMQ-based communication between ego clients and the
open-world server for:
- Registration and ID assignment (REQ-REP)
- State queries (REQ-REP)
- Event subscription (PUB-SUB)
- Ready signaling (REQ-REP)

Only used in open-world multi-client mode, not needed for training.
"""

import zmq
import yaml
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Callable

logger = logging.getLogger(__name__)


class ServerClient:
    """
    ZeroMQ client for communicating with open-world server.

    Handles:
    - Registration and ID assignment via REQ-REP socket
    - State queries via REQ-REP socket
    - Event subscription via PUB-SUB socket
    - Ready signaling via REQ-REP socket
    """

    def __init__(self, host: str = 'localhost', config_path: Optional[Path] = None):
        """
        Initialize server client with ZeroMQ sockets.

        Args:
            host: Server host address
            config_path: Path to carla_config.yaml (auto-detected if None)
        """
        self.host = host

        # Load configuration to get ports
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent.parent / "config" / "carla_config.yaml"

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            server_config = config.get('server', {})
            self.req_port = server_config.get('zeromq_rep_port', 5555)
            self.pub_port = server_config.get('zeromq_pub_port', 5556)
            logger.info(f"Loaded ZeroMQ ports from config: REP={self.req_port}, PUB={self.pub_port}")
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}. Using default ports.")
            self.req_port = 5555
            self.pub_port = 5556

        # Create ZeroMQ context
        self.context = zmq.Context()

        # REQ socket for queries (request-reply pattern)
        self.req_socket = self.context.socket(zmq.REQ)
        self.req_socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10 second receive timeout
        self.req_socket.setsockopt(zmq.SNDTIMEO, 10000)  # 10 second send timeout
        self.req_socket.connect(f"tcp://{host}:{self.req_port}")
        logger.info(f"Connected REQ socket to tcp://{host}:{self.req_port}")

        # SUB socket for events (publish-subscribe pattern)
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{host}:{self.pub_port}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
        logger.info(f"Connected SUB socket to tcp://{host}:{self.pub_port}")

        # Validate connection by testing with get_state
        try:
            self.req_socket.send_json({"type": "get_state"})
            response = self.req_socket.recv_json()
            logger.info(f"Connection validated - server phase: {response.get('phase')}")
        except zmq.Again:
            logger.error(f"Connection validation failed - server not responding at {host}:{self.req_port}")
            self.cleanup()
            raise TimeoutError(f"Server not responding at {host}:{self.req_port}")
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            self.cleanup()
            raise

        # Shutdown handling
        self.shutdown_received = False
        self.go_received = threading.Event()  # Event to signal when GO received
        self.on_shutdown_callback = None  # Optional callback when shutdown received
        self.assigned_client_id = None  # Set after registration

        # Start background listener for server events
        self.listener_running = True
        self.listener_thread = threading.Thread(target=self._listen_for_events, daemon=True)
        self.listener_thread.start()

    def register(self) -> Dict:
        """
        Register with server and get assigned client_id.

        This is an explicit resource allocation operation that assigns
        a unique client ID. Should only be called once per client.

        Returns:
            Dictionary with registration info:
            {
                "assigned_client_id": 0 | 1 | 2 | ...,
                "phase": "waiting" | "running"
            }
        """
        try:
            self.req_socket.send_json({"type": "register"})
            response = self.req_socket.recv_json()

            # Store client_id for cleanup
            self.assigned_client_id = response.get('assigned_client_id')

            logger.info(f"Registered with server: client_id={self.assigned_client_id}")
            return response
        except zmq.Again:
            logger.error("Timeout registering with server - is the server running?")
            raise TimeoutError("Server registration timeout")
        except Exception as e:
            logger.error(f"Failed to register with server: {e}")
            raise

    def get_state(self) -> Dict:
        """
        Query current server state (read-only operation, no side effects).

        Can be called multiple times without changing server state.

        Returns:
            Dictionary with state information:
            {
                "phase": "waiting" | "running",
                "carla_host": "...",
                "carla_port": 2000
            }
        """
        try:
            self.req_socket.send_json({"type": "get_state"})
            state = self.req_socket.recv_json()
            logger.debug(f"Server state: {state}")
            return state
        except zmq.Again:
            logger.error("Timeout querying server state - is the server running?")
            raise TimeoutError("Server state query timeout")
        except Exception as e:
            logger.error(f"Failed to get server state: {e}")
            raise

    def signal_ready(self, client_id: int) -> Dict:
        """
        Signal to server that client is ready to start.

        Args:
            client_id: Client ID assigned by server (from get_state)

        Returns:
            Server acknowledgment response
        """
        try:
            self.req_socket.send_json({
                "type": "ready",
                "client_id": client_id
            })
            response = self.req_socket.recv_json()
            logger.info(f"Server acknowledged ready signal for client {client_id}")
            return response
        except zmq.Again:
            logger.error("Timeout signaling ready to server")
            raise TimeoutError("Server ready signal timeout")
        except Exception as e:
            logger.error(f"Failed to signal ready: {e}")
            raise

    def wait_for_go(self, timeout: Optional[float] = None) -> bool:
        """
        Block until GO event is received from server.

        Uses the background listener thread to receive events.

        Args:
            timeout: Optional timeout in seconds (None = wait forever)

        Returns:
            True if GO received, False if timeout
        """
        logger.info("Waiting for GO signal from server...")

        # Wait for the event to be set by listener thread
        received = self.go_received.wait(timeout=timeout)

        if received:
            logger.info("GO signal received - starting!")
            return True
        else:
            logger.warning("Timeout waiting for GO signal")
            return False

    def _listen_for_events(self):
        """Background thread listening for server events (GO, SHUTDOWN)."""
        import time
        while self.listener_running:
            try:
                event = self.sub_socket.recv_json(flags=zmq.NOBLOCK)

                event_type = event.get("event")

                if event_type == "GO":
                    logger.info("GO signal received from server!")
                    self.go_received.set()  # Signal waiting threads

                elif event_type == "SHUTDOWN":
                    logger.warning("Server shutdown signal received!")
                    self.shutdown_received = True

                    # Trigger callback if registered
                    if self.on_shutdown_callback:
                        try:
                            self.on_shutdown_callback()
                        except Exception as e:
                            logger.error(f"Shutdown callback error: {e}")

                    # Stop listening
                    break

            except zmq.Again:
                # No message available - sleep briefly
                time.sleep(0.01)
            except Exception as e:
                if self.listener_running:
                    logger.error(f"Event listener error: {e}")
                break

        logger.debug("Event listener thread stopped")

    def request_cleanup(self, ego_actor_id):
        """Request server to clean up client's actors."""
        if self.assigned_client_id is None:
            return

        try:
            self.req_socket.send_json({
                "type": "cleanup_request",
                "client_id": self.assigned_client_id,
                "ego_actor_id": ego_actor_id
            })
            response = self.req_socket.recv_json()
            logger.info(f"Server cleanup complete: {response.get('status')}")
        except Exception as e:
            logger.warning(f"Cleanup request failed: {e}")

    def disconnect(self):
        """Notify server of clean disconnect."""
        if self.assigned_client_id is None:
            logger.debug("No client_id - skipping disconnect notification")
            return

        try:
            logger.info(f"Disconnecting client {self.assigned_client_id}...")
            self.req_socket.send_json({
                "type": "disconnect",
                "client_id": self.assigned_client_id
            })
            response = self.req_socket.recv_json()
            logger.info("Disconnected from server")
        except Exception as e:
            logger.warning(f"Disconnect notification failed: {e}")

    def cleanup(self):
        """Close ZeroMQ sockets and context."""
        try:
            # Stop listener thread
            self.listener_running = False

            # Wait for listener thread to finish
            if hasattr(self, 'listener_thread') and self.listener_thread.is_alive():
                logger.debug("Waiting for listener thread to stop...")
                self.listener_thread.join(timeout=1.0)

            # Notify server of disconnect
            self.disconnect()

            logger.info("Closing ZeroMQ sockets...")
            self.req_socket.close()
            self.sub_socket.close()
            self.context.term()
            logger.info("ZeroMQ cleanup complete")
        except Exception as e:
            logger.warning(f"Error during ZeroMQ cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup on exit."""
        self.cleanup()
        return False
