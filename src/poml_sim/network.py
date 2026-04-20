"""Simulated P2P network for PoML simulation.

NetworkBus holds per-miner queues and simulates propagation delay.
"""

from __future__ import annotations

import logging
import multiprocessing
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from queue import Empty
from typing import Any

logger = logging.getLogger(__name__)


class MessageType(Enum):
    NEW_BLOCK = auto()
    NEW_QUERY = auto()
    STOP = auto()


@dataclass
class NetworkMessage:
    msg_type: MessageType
    sender_id: int
    payload: Any
    timestamp: float = 0.0


class NetworkBus:
    """Simulated P2P network with configurable latency."""

    def __init__(self, num_miners: int, latency_ms: int = 100) -> None:
        self.num_miners = num_miners
        self.latency_s = latency_ms / 1000.0
        # Per-miner inbox queues
        self.inboxes: dict[int, multiprocessing.Queue[NetworkMessage]] = {
            i: multiprocessing.Queue() for i in range(num_miners)
        }

    def broadcast(self, message: NetworkMessage, sender_id: int) -> None:
        """Send message to all miners except sender, with simulated delay."""
        for miner_id, inbox in self.inboxes.items():
            if miner_id == sender_id:
                continue
            if self.latency_s > 0:
                t = threading.Timer(
                    self.latency_s,
                    inbox.put,
                    args=(message,),
                )
                t.daemon = True
                t.start()
            else:
                inbox.put(message)

    def send_to(self, miner_id: int, message: NetworkMessage) -> None:
        """Send a message directly to a specific miner."""
        self.inboxes[miner_id].put(message)

    def broadcast_all(self, message: NetworkMessage) -> None:
        """Send message to ALL miners (including sender), with delay."""
        for miner_id, inbox in self.inboxes.items():
            if self.latency_s > 0:
                t = threading.Timer(
                    self.latency_s,
                    inbox.put,
                    args=(message,),
                )
                t.daemon = True
                t.start()
            else:
                inbox.put(message)

    def get_messages(self, miner_id: int, timeout: float = 0.0) -> list[NetworkMessage]:
        """Get all pending messages for a miner (non-blocking by default)."""
        messages: list[NetworkMessage] = []
        inbox = self.inboxes[miner_id]
        try:
            if timeout > 0:
                msg = inbox.get(timeout=timeout)
                messages.append(msg)
            # Drain remaining
            while True:
                msg = inbox.get_nowait()
                messages.append(msg)
        except Empty:
            pass
        return messages

    def stop_all(self) -> None:
        """Send STOP to all miners."""
        stop_msg = NetworkMessage(
            msg_type=MessageType.STOP,
            sender_id=-1,
            payload=None,
            timestamp=time.time(),
        )
        for inbox in self.inboxes.values():
            inbox.put(stop_msg)
