from redis import ConnectionPool, Redis, Sentinel, SSLConnection

from grimoire.core.settings import settings


class RedisManager:
    def __init__(self, host: list[tuple[str, int]], sentinel=False, sentinel_master="", tls=False) -> None:
        self.sentinel = sentinel
        self.host = host
        self.tls = tls
        self.sentinel_master = sentinel_master
        if sentinel:
            self._init_sentinel()
        else:
            self._init_single()

    def _init_sentinel(self) -> None:
        if self.tls:
            self.sentinel_client = Sentinel(self.host, decode_responses=True, ssl=True, ssl_cert_reqs="none")
        else:
            self.sentinel_client = Sentinel(self.host, decode_responses=True)

    def _init_single(self) -> None:
        if self.tls:
            self.connection_pool = ConnectionPool(
                host=self.host[0][0],
                port=self.host[0][1],
                db=0,
                decode_responses=True,
                connection_class=SSLConnection,
                ssl_cert_reqs="none",
            )
        else:
            self.connection_pool = ConnectionPool(host=self.host[0][0], port=self.host[0][1], db=0, decode_responses=True)

    def get_client(self) -> Redis:
        if self.sentinel:
            return self.sentinel_client.master_for(self.sentinel_master)
        else:
            return Redis(connection_pool=self.connection_pool)

    def celery_broker_url(self) -> str:
        if self.sentinel:
            connection_strings = [
                f"sentinel://{host_name}:{port}" for host_name, port in self.host
            ]
            return ";".join(connection_strings)

        else:
            if self.tls:
                return f"rediss://{self.host[0][0]}:{self.host[0][1]}/0?ssl_cert_reqs=none"
            else:
                return f"redis://{self.host[0][0]}:{self.host[0][1]}/0"


redis_manager = RedisManager(
    settings.redis.HOST,
    settings.redis.SENTINEL,
    settings.redis.SENTINEL_MASTER_NAME,
    settings.redis.TLS,
)
