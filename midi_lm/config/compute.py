from dataclasses import dataclass


@dataclass
class LocalConfig:
    local: bool = True


@dataclass
class ModalConfig:
    cpu: int
    memory: int
    gpu: str | None
    timeout: int = 60 * 60 * 12
    local: bool = False


@dataclass
class ModalCpuConfig(ModalConfig):
    cpu: int = 4
    memory: int = 3814  # 4gb
    gpu: str | None = None


@dataclass
class ModalGpuA10gConfig(ModalConfig):
    cpu: int = 4
    memory: int = 3814  # 4gb
    gpu: str | None = "a10g"


@dataclass
class ModalGpuA100Config(ModalConfig):
    cpu: int = 8
    memory: int = 7628  # 8gb
    gpu: str | None = "a100"
