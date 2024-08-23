from redis import ConnectionPool, Redis, Sentinel, SSLConnection

from grimoire.core.settings import settings


class RedisManager:
    def __init__(self, host: str, port: str | int, sentinel=False, sentinel_master="", tls=False) -> None:
        self.sentinel = sentinel
        self.host = host
        self.port = port
        self.tls = tls
        self.sentinel_master = sentinel_master
        if sentinel:
            self._init_sentinel()
        else:
            self._init_single()

    def _init_sentinel(self) -> None:
        hosts = self.host.split(",")
        ports = self.port.split(",")
        sentinel_hosts = [(host_name, int(port)) for host_name, port in zip(hosts, ports, strict=True)]

        if self.tls:
            self.sentinel_client = Sentinel(sentinel_hosts, decode_responses=True, ssl=True, ssl_cert_reqs="none")
        else:
            self.sentinel_client = Sentinel(sentinel_hosts, decode_responses=True)

    def _init_single(self) -> None:
        if self.tls:
            self.connection_pool = ConnectionPool(
                self.host,
                int(self.port),
                db=0,
                decode_responses=True,
                connection_class=SSLConnection,
                ssl_cert_reqs="none",
            )
        else:
            self.connection_pool = ConnectionPool(self.host, int(self.port), db=0, decode_responses=True)

    def get_client(self) -> Redis:
        if self.sentinel:
            return self.sentinel_client.master_for(self.sentinel_master)
        else:
            return Redis(connection_pool=self.connection_pool)

    def celery_broker_url(self) -> str:
        if self.sentinel:
            hosts = self.host.split(",")
            ports = self.port.split(",")
            connection_strings = [
                f"sentinel://{host_name}:{port}" for host_name, port in zip(hosts, ports, strict=True)
            ]
            return ";".join(connection_strings)

        else:
            if self.tls:
                return f"rediss://{self.host}:{self.port}/0?ssl_cert_reqs=none"
            else:
                return f"redis://{self.host}:{self.port}/0"


redis_manager = RedisManager(
    settings["REDIS_HOST"],
    settings["REDIS_PORT"],
    settings["REDIS_SENTINEL"],
    settings["SENTINEL_MASTER_NAME"],
    settings["REDIS_TLS"],
)
