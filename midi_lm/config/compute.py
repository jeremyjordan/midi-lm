from dataclasses import dataclass


@dataclass
class LocalConfig:
    local: bool = True


@dataclass
class ModalCpuConfig:
    local: bool = False
    hardware: str = "cpu"


@dataclass
class ModalGpuA10gConfig:
    local: bool = False
    hardware: str = "a10g"


@dataclass
class ModalGpuA100Config:
    local: bool = False
    hardware: str = "a100"


@dataclass
class ModalGpuH100Config:
    local: bool = False
    hardware: str = "h100"
