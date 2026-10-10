# mission_sim/simulation/hitl.py
"""
MCPC L4: HITL/SITL 接口预留（契约 + 时间步握手，mock 可测）
------------------------------------------------------------
按 DD-001 R-L4-05：先契约后模型。消息信封 + 握手协议 + 超时处理；
配套 `MockEchoTransport` 供离线测试（真实 HITL 硬件接入由 L4+ 实施）。

协议（v1）：
    hello      → hello_ack（回显 protocol_version / sim_time）
    state      → actuator_cmd（载荷 AttitudeState，回发执行指令）
"""

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

from mission_sim.core.physics.ids import AttitudeState

PROTOCOL_VERSION = 1


class HITLError(RuntimeError):
    """HITL 通用错误。"""


class HITLHandshakeError(HITLError):
    """握手失败（版本不匹配 / 协议错误）。"""


class HITLTimeoutError(HITLError):
    """等待应答超时（链路失效）。"""


@dataclass
class HITLMessage:
    """HITL 消息信封（可序列化往返）。"""
    msg_type: str
    sim_time: float
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": self.msg_type, "sim_time": float(self.sim_time),
                "payload": self.payload}

    @classmethod
    def from_dict(cls, d: dict) -> "HITLMessage":
        return cls(msg_type=d["type"], sim_time=float(d["sim_time"]),
                   payload=dict(d.get("payload", {})))


class HITLTransport(Protocol):
    def send(self, msg: HITLMessage) -> None: ...
    def receive(self) -> Optional[HITLMessage]: ...


class MockEchoTransport:
    """离线 mock：对 hello/state 立即回显对应应答；可配置版本错误或断链。"""

    def __init__(self, reply_version: int = PROTOCOL_VERSION, dead: bool = False):
        self.reply_version = int(reply_version)
        self.dead = bool(dead)
        self.sent = []

    def send(self, msg: HITLMessage) -> None:
        self.sent.append(msg)

    def receive(self) -> Optional[HITLMessage]:
        if self.dead or not self.sent:
            return None
        m = self.sent[-1]
        if m.msg_type == "hello":
            return HITLMessage("hello_ack", m.sim_time,
                               {"protocol_version": self.reply_version,
                                "echo_sim_time": m.sim_time})
        if m.msg_type == "state":
            return HITLMessage("actuator_cmd", m.sim_time, {"command": [0.0, 0.0, 0.0]})
        return None


class HITLInterface:
    """HITL 仿真侧接口：握手 + 时间步交换 + 超时处理。"""

    PROTOCOL_VERSION = PROTOCOL_VERSION

    def __init__(self, transport: HITLTransport, dt: float, timeout_steps: int = 5):
        self.transport = transport
        self.dt = float(dt)
        self.timeout_steps = int(timeout_steps)

    def _poll(self, expect_type: str, timeout_steps: int) -> HITLMessage:
        for _ in range(timeout_steps):
            r = self.transport.receive()
            if r is None:
                continue
            if r.msg_type != expect_type:
                raise HITLHandshakeError(
                    f"协议错误：期望 {expect_type}，收到 {r.msg_type}")
            return r
        raise HITLTimeoutError(f"等待 {expect_type} 超时（{timeout_steps} 步）")

    def handshake(self, timeout_steps: Optional[int] = None) -> dict:
        """发送 hello 并校验 hello_ack（版本 + 时间回显）。"""
        self.transport.send(HITLMessage("hello", 0.0,
                                        {"protocol_version": self.PROTOCOL_VERSION}))
        r = self._poll("hello_ack", timeout_steps or self.timeout_steps)
        version = int(r.payload.get("protocol_version", -1))
        if version != self.PROTOCOL_VERSION:
            raise HITLHandshakeError(
                f"协议版本不匹配：对端 {version}，本端 {self.PROTOCOL_VERSION}")
        return {"protocol_version": version, "echo_sim_time": r.payload.get("echo_sim_time")}

    def step(self, epoch: float, attitude: AttitudeState,
             timeout_steps: Optional[int] = None) -> np.ndarray:
        """时间步握手：发送 state（含 AttitudeState），返回执行指令 (3,)。"""
        self.transport.send(HITLMessage("state", float(epoch),
                                        {"attitude": attitude.as_dict()}))
        r = self._poll("actuator_cmd", timeout_steps or self.timeout_steps)
        return np.asarray(r.payload.get("command", [0.0, 0.0, 0.0]), dtype=np.float64)
