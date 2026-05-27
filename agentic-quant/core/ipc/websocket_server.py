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
from core.memory.short_term.redis_cache_manager import RedisCacheManager
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
        redis_cache: RedisCacheManager | None = None,
    ) -> None:
        self._host = host
        self._port_start = port_start
        self._port_end = port_end
        self._actual_port: int | None = None
        self._server: Any = None  # websockets server
        self._clients: set[ServerConnection] = set()
        self._broadcast_dispatcher: BroadcastDispatcher | None = None
        self._stop_event = asyncio.Event()
        self._redis_cache: RedisCacheManager | None = redis_cache

        logger.info(
            f"WebSocketServer khoi tao: {host}:{port_start}-{port_end} "
            f"(redis={'co' if redis_cache else 'khong'})"
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

    async def _build_full_state(self) -> FullStateSnapshotMessage:
        """Xay dung FullStateSnapshotMessage tu Redis/SQLite.

        Thu lay du lieu tu Redis truoc:
          - bars: {tf: [ohlcv]} — lay 500 bar gan nhat moi TF
          - zones: list[dict] — zone dang active
          - predictions: dict — AI prediction moi nhat
          - consensus: dict — debate/consensus record
          - macro_context: dict — macro state + events
          - model_status: str — trang thai model (healthy/degraded)

        Fallback: neu Redis ko available, tra ve empty snapshot
        voi system status = "running" + warning logged.

        Returns:
            FullStateSnapshotMessage: snapshot hien tai
        """
        bars: list[dict[str, Any]] = []
        zones: list[dict[str, Any]] = []
        predictions: dict[str, Any] = {}
        consensus: dict[str, Any] = {}
        macro_context: dict[str, Any] = {}
        model_status: str = "healthy"

        if self._redis_cache is not None and self._redis_cache.is_connected:
            try:
                redis = self._redis_cache.client

                # --- Bars: scan tung TF, lay 500 bar moi nhat ---
                timeframes = ["M1", "M5", "M15", "H1", "H4", "D1"]
                for tf in timeframes:
                    try:
                        bar_keys = []
                        async for key in redis.scan_iter(
                            match=f"bar:{tf}:*", count=500
                        ):
                            key_str = key.decode() if isinstance(key, bytes) else key
                            bar_keys.append(key_str)

                        if not bar_keys:
                            continue

                        # Lay bars, sort theo thoi gian va gioi han 500
                        for bk in sorted(bar_keys)[:500]:
                            raw = await redis.hgetall(bk)
                            if raw:
                                decoded = {
                                    k.decode(): self._decode_redis_val(v)
                                    for k, v in raw.items()
                                }
                                decoded["tf"] = tf
                                bars.append(decoded)
                    except Exception as e:
                        logger.warning(
                            f"Loi lay bars cho TF {tf}: {e}"
                        )
                        continue

                # --- Zones: quet zone keys ---
                try:
                    zone_keys = []
                    async for key in redis.scan_iter(
                        match=f"zone:*", count=1000
                    ):
                        zone_keys.append(
                            key.decode() if isinstance(key, bytes) else key
                        )
                    for zk in zone_keys:
                        raw = await redis.hgetall(zk)
                        if raw:
                            decoded = {
                                k.decode(): self._decode_redis_val(v)
                                for k, v in raw.items()
                            }
                            zones.append(decoded)
                except Exception as e:
                    logger.warning(f"Loi lay zones tu Redis: {e}")

                # --- Predictions (AI Output) ---
                try:
                    # Doc prediction cho symbol mac dinh
                    for symbol in ["XAUUSD", "EURUSD"]:
                        pred_key = f"ai:output:{symbol}:latest"
                        raw = await redis.hgetall(pred_key)
                        if raw:
                            decoded = {
                                k.decode(): self._decode_redis_val(v)
                                for k, v in raw.items()
                            }
                            predictions[symbol] = decoded
                except Exception as e:
                    logger.warning(
                        f"Loi lay predictions tu Redis: {e}"
                    )

                # --- Consensus / Debate ---
                try:
                    async for key in redis.scan_iter(
                        match="debate:*", count=100
                    ):
                        key_str = (
                            key.decode() if isinstance(key, bytes) else key
                        )
                        raw = await redis.hgetall(key_str)
                        if raw:
                            decoded = {
                                k.decode(): self._decode_redis_val(v)
                                for k, v in raw.items()
                            }
                            consensus[key_str] = decoded
                except Exception as e:
                    logger.warning(
                        f"Loi lay consensus tu Redis: {e}"
                    )

                # --- Macro State ---
                try:
                    async for key in redis.scan_iter(
                        match="macro:state:*", count=50
                    ):
                        key_str = (
                            key.decode() if isinstance(key, bytes) else key
                        )
                        raw = await redis.hgetall(key_str)
                        if raw:
                            decoded = {
                                k.decode(): self._decode_redis_val(v)
                                for k, v in raw.items()
                            }
                            currency = key_str.split(":")[-1]
                            macro_context[currency] = decoded
                except Exception as e:
                    logger.warning(
                        f"Loi lay macro state tu Redis: {e}"
                    )

                # --- Macro Events ---
                try:
                    async for key in redis.scan_iter(
                        match="macro:events:*", count=50
                    ):
                        key_str = (
                            key.decode() if isinstance(key, bytes) else key
                        )
                        events_raw = await redis.lrange(
                            key_str, 0, -1
                        )
                        if events_raw:
                            currency = key_str.split(":")[1]
                            events = []
                            for ev_bytes in events_raw:
                                try:
                                    ev = json.loads(
                                        ev_bytes.decode("utf-8")
                                    )
                                    events.append(ev)
                                except Exception:
                                    pass
                            if "events" not in macro_context:
                                macro_context["events"] = {}
                            macro_context["events"][currency] = events
                except Exception as e:
                    logger.warning(
                        f"Loi lay macro events tu Redis: {e}"
                    )

                # --- Model Status ---
                try:
                    metrics_key = "metrics:model:latest"
                    raw = await redis.hgetall(metrics_key)
                    if raw:
                        decoded = {
                            k.decode(): self._decode_redis_val(v)
                            for k, v in raw.items()
                        }
                        if decoded.get("degraded", "false") == "true":
                            model_status = "degraded"
                except Exception as e:
                    logger.warning(
                        f"Loi lay model status tu Redis: {e}"
                    )

            except Exception as e:
                logger.warning(
                    f"Redis unavailable for full state build: {e}"
                )
                # Fallback: tra ve empty state
                bars = []
                zones = []
                predictions = {}
                consensus = {}
                macro_context = {}
                model_status = "unknown"
        else:
            logger.info(
                "Redis not configured/connected — building empty full state"
            )

        return FullStateSnapshotMessage(
            symbol="XAUUSD",
            bars=bars,
            zones=zones,
            predictions=predictions,
            consensus=consensus,
            system={
                "status": "running",
                "model_status": model_status,
                "bar_count": len(bars),
                "zone_count": len(zones),
                "redis_connected": (
                    self._redis_cache.is_connected
                    if self._redis_cache
                    else False
                ),
            },
            timestamp=int(time.time()),
        )

    async def _send_full_state(
        self, websocket: ServerConnection
    ) -> None:
        """Gui FullStateSnapshotMessage cho client.

        Goi _build_full_state() de thu thap trang thai hien tai
        tu Redis, sau do serialize va gui qua WebSocket.
        """
        msg = await self._build_full_state()
        await self._send_json(websocket, msg.model_dump())
        logger.debug(
            f"Da gui full_state_snapshot cho client "
            f"(bars={len(msg.bars)}, zones={len(msg.zones)}, "
            f"predictions={len(msg.predictions)}, "
            f"system_status={msg.system.get('status', 'unknown')})"
        )

    @staticmethod
    def _decode_redis_val(value: Any) -> Any:
        """Giai ma gia tri tu Redis (bytes -> int/float/str).

        Args:
            value: Gia tri tu Redis (bytes, int, float, str)

        Returns:
            Gia tri da decode
        """
        if isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8")
                # Thu ep kieu so
                try:
                    if "." in decoded:
                        return float(decoded)
                    return int(decoded)
                except (ValueError, TypeError):
                    return decoded
            except UnicodeDecodeError:
                return str(value)
        return value

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
