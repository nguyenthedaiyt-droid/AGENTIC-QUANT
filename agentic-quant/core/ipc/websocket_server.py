# =============================================================================
# AGENTIC-QUANT — WebSocket IPC Server
# Server WebSocket async de giao tiep voi Frontend (Tauri/React)
# =============================================================================

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import websockets
from loguru import logger
from websockets.asyncio.server import ServerConnection, serve

from core.ipc.broadcast_dispatcher import BroadcastDispatcher
from core.ipc.message_schema import (
    BarUpdateMessage,
    CountdownUpdateMessage,
    FullStateSnapshotMessage,
)
from core.utils.events import get_event_bus
from core.utils.events.types import EventType

if TYPE_CHECKING:
    from collections.abc import Callable, Set as AbstractSet


# =============================================================================
# WebSocketServer
# =============================================================================

class WebSocketServer:
    """Server WebSocket IPC de push data tu Backend -> Frontend.

    - start() -> port (try 47290, fallback 47291-47299)
    - Ghi port vao /tmp/aq_ws_port.txt
    - Print 'AGENTIQ_BACKEND_READY' sau khi start thanh cong
    - _handler(websocket): manage clients, handle 'request_full_state'
    - Subscribe vao BroadcastDispatcher de broadcast
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port_start: int = 47290,
        port_end: int = 47299,
    ) -> None:
        self._host = host
        self._port_start = port_start
        self._port_end = port_end
        self._actual_port: int | None = None
        self._server: Any = None  # websockets server
        self._clients: set[ServerConnection] = set()
        self._broadcast_dispatcher: BroadcastDispatcher | None = None
        self._stop_event = asyncio.Event()

        logger.info(
            f"WebSocketServer khoi tao: {host}:{port_start}-{port_end}"
        )

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    async def start(self) -> int:
        """Khoi dong WebSocket server voi port fallback.

        Returns:
            port dang chay
        """
        event_bus = await get_event_bus()

        # Tao BroadcastDispatcher va gan broadcast fn
        self._broadcast_dispatcher = BroadcastDispatcher(event_bus)
        self._broadcast_dispatcher.set_broadcast_fn(self._broadcast_to_all)

        # Thu port tu start -> end
        for port in range(self._port_start, self._port_end + 1):
            try:
                self._server = await serve(
                    self._handler,
                    self._host,
                    port,
                )
                self._actual_port = port
                logger.info(f"WebSocket server listening on {self._host}:{port}")
                break
            except OSError as e:
                logger.warning(
                    f"Port {port} khong kha dung ({e}), thu port tiep..."
                )
                continue

        if self._actual_port is None:
            raise RuntimeError(
                f"Khong the bind WebSocket server tren bat ky port nao "
                f"trong range {self._port_start}-{self._port_end}"
            )

        # Ghi port ra file /tmp/aq_ws_port.txt
        port_file = Path("/tmp/aq_ws_port.txt")
        port_file.write_text(str(self._actual_port))
        logger.info(f"Da ghi port {self._actual_port} vao {port_file}")

        # Khoi dong BroadcastDispatcher subscribe vao EventBus
        self._broadcast_dispatcher.start()

        # Print ready signal cho Tauri doc
        print("AGENTIQ_BACKEND_READY", flush=True)
        logger.info("AGENTIQ_BACKEND_READY da gui toi stdout")

        # Start heartbeat task
        asyncio.create_task(self._heartbeat_loop())

        return self._actual_port

    async def stop(self) -> None:
        """Dung WebSocket server va cleanup."""
        self._stop_event.set()

        if self._broadcast_dispatcher is not None:
            self._broadcast_dispatcher.stop()

        # Dong tat ca client connections
        for ws in list(self._clients):
            try:
                await ws.close(1001, "Server shutting down")
            except Exception:
                pass
        self._clients.clear()

        # Dong server
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

        logger.info("WebSocket server da dung")

    # ------------------------------------------------------------------
    # WebSocket Handler
    # ------------------------------------------------------------------

    async def _handler(
        self, websocket: ServerConnection
    ) -> None:
        """Handler cho moi client ket noi.

        - Them vao clients set
        - Lang nghe incoming messages (text)
        - Xu ly 'request_full_state'
        - Cleanup khi disconnect
        """
        # Them client
        self._clients.add(websocket)
        remote = websocket.remote_address
        logger.info(
            f"WebSocket client ket noi: {remote} "
            f"(total: {len(self._clients)})"
        )

        try:
            async for raw_message in websocket:
                try:
                    data = json.loads(raw_message)
                    await self._handle_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Invalid JSON tu client {remote}: {raw_message[:200]}"
                    )
                except Exception:
                    logger.exception(
                        f"Loi xu ly message tu {remote}"
                    )
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket client ngat ket noi: {remote}")
        except Exception:
            logger.exception(f"Loi WebSocket handler cho {remote}")
        finally:
            self._clients.discard(websocket)
            logger.info(
                f"WebSocket client disconnect: {remote} "
                f"(total: {len(self._clients)})"
            )

    async def _handle_message(
        self, websocket: ServerConnection, data: dict[str, Any]
    ) -> None:
        """Xu ly incoming message tu client."""
        msg_type = data.get("type", "")

        if msg_type == "request_full_state":
            await self._send_full_state(websocket)
        else:
            logger.debug(f"Unknown message type tu client: {msg_type}")

    async def _send_full_state(
        self, websocket: ServerConnection
    ) -> None:
        """Gui FullStateSnapshotMessage cho client.

        TODO: Lay du lieu tu Redis/cache de fill snapshot.
        Hien tai gui snapshot voi du lieu empty.
        """
        msg = FullStateSnapshotMessage(
            system={
                "status": "running",
                "uptime_ms": 0,
            },
            timestamp=int(time.time()),
        )
        await self._send_json(websocket, msg.model_dump())
        logger.debug("Da gui full_state_snapshot cho client")

    # ------------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------------

    def _broadcast_to_all(self, message: dict[str, Any]) -> None:
        """Broadcast JSON message den tat ca connected clients.

        Chay synchronous de goi tu BroadcastDispatcher callback.
        Tao asyncio task de khong blocking handler.
        """
        if not self._clients:
            return

        # Gui batch toi tat ca clients
        payload = json.dumps(message, default=str)
        for ws in list(self._clients):
            try:
                asyncio.create_task(self._send_raw(ws, payload))
            except Exception:
                logger.exception("Loi broadcast to client")
                self._clients.discard(ws)

    async def _send_json(
        self, websocket: ServerConnection, message: dict[str, Any]
    ) -> None:
        """Gui JSON dict toi mot client cu the."""
        try:
            payload = json.dumps(message, default=str)
            await websocket.send(payload)
        except websockets.exceptions.ConnectionClosed:
            self._clients.discard(websocket)
        except Exception:
            logger.exception("Loi send_json")
            self._clients.discard(websocket)

    async def _send_raw(
        self, websocket: ServerConnection, payload: str
    ) -> None:
        """Gui raw JSON string toi mot client."""
        try:
            await websocket.send(payload)
        except websockets.exceptions.ConnectionClosed:
            self._clients.discard(websocket)
        except Exception:
            logger.exception("Loi send_raw")
            self._clients.discard(websocket)

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Gui heartbeat ping de giu connection alive."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(30)
                if not self._clients:
                    continue
                heartbeat = {
                    "type": "heartbeat",
                    "emit_time_ms": int(time.time() * 1000),
                }
                payload = json.dumps(heartbeat)
                for ws in list(self._clients):
                    try:
                        await ws.send(payload)
                    except websockets.exceptions.ConnectionClosed:
                        self._clients.discard(ws)
                    except Exception:
                        self._clients.discard(ws)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Loi heartbeat loop")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def port(self) -> int | None:
        """Port dang chay."""
        return self._actual_port

    @property
    def client_count(self) -> int:
        """So luong client dang ket noi."""
        return len(self._clients)
