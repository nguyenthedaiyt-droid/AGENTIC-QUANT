# =============================================================================
# AGENTIC-QUANT — IPC Module (WebSocket Server)
# Cung cap WebSocket IPC server, BroadcastDispatcher va Message Schema
# =============================================================================
"""
IPC Module: WebSocket server cho giao tiep Backend -> Frontend.

Su dung::

    from core.ipc import WebSocketServer, BroadcastDispatcher
    from core.ipc.message_schema import BarUpdateMessage

    ws_server = WebSocketServer()
    port = await ws_server.start()
"""

from .broadcast_dispatcher import BroadcastDispatcher
from .message_schema import (
    AllIPCMessages,
    BarUpdateMessage,
    BaseIPCMessage,
    ConsensusReadyMessage,
    CountdownUpdateMessage,
    FullStateSnapshotMessage,
    NewsReleaseMessage,
    PredictionUpdateMessage,
    SystemStatusMessage,
    ZoneUpdateMessage,
)
from .websocket_server import WebSocketServer

__all__ = [
    # Server
    "WebSocketServer",
    "BroadcastDispatcher",
    # Message Types
    "BaseIPCMessage",
    "BarUpdateMessage",
    "ZoneUpdateMessage",
    "PredictionUpdateMessage",
    "CountdownUpdateMessage",
    "ConsensusReadyMessage",
    "SystemStatusMessage",
    "FullStateSnapshotMessage",
    "NewsReleaseMessage",
    "AllIPCMessages",
]
