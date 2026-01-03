"""Configuration management for the multi-agent system."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"

    @classmethod
    def from_env(cls) -> "RedisConfig":
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
        )


@dataclass
class BrokerConfig:
    """Message broker configuration."""
    redis: RedisConfig = field(default_factory=RedisConfig)
    channel_prefix: str = "lida"
    broadcast_channel: str = "broadcast"
    dead_letter_channel: str = "dead_letters"
    message_ttl_seconds: int = 3600
    max_retries: int = 3
    ack_timeout_seconds: float = 5.0
    batch_size: int = 100


@dataclass
class MailboxConfig:
    """Mailbox configuration."""
    max_size: int = 10000
    high_priority_ratio: float = 0.3
    rate_limit_per_second: int = 1000
    enable_deduplication: bool = True
    dedup_window_seconds: int = 60


@dataclass
class AgentConfig:
    """Agent configuration."""
    agent_id: Optional[str] = None
    agent_type: str = "generic"
    role: str = "worker"
    mailbox: MailboxConfig = field(default_factory=MailboxConfig)
    heartbeat_interval_seconds: float = 30.0
    shutdown_timeout_seconds: float = 10.0
    max_concurrent_tasks: int = 10
    system_prompt: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SupervisorConfig:
    """Supervisor configuration."""
    max_restarts: int = 3
    restart_window_seconds: float = 60.0
    restart_delay_seconds: float = 1.0
    health_check_interval_seconds: float = 10.0


@dataclass
class UIConfig:
    """UI configuration."""
    enable_colors: bool = True
    refresh_rate_hz: float = 4.0
    max_log_lines: int = 100
    show_message_content: bool = False  # Privacy
    truncate_length: int = 50


@dataclass
class Config:
    """Root configuration."""
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    debug: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            broker=BrokerConfig(redis=RedisConfig.from_env()),
            debug=os.getenv("DEBUG", "").lower() in ("1", "true", "yes"),
        )
