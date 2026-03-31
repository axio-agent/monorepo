import logging

import argclass
from textual.logging import TextualHandler


class LogGroup(argclass.Group):
    """Logging options."""

    level: str = argclass.LogLevel  # type: ignore[assignment]
    file: str = argclass.Argument(default=None, metavar="PATH", help="log file path")

    def configure(self) -> None:
        handlers: list[logging.Handler] = [TextualHandler()]
        if self.file:
            handlers.append(logging.FileHandler(self.file, mode="a"))

        logging.basicConfig(
            level=self.level,
            format="%(asctime)s %(levelname)s[%(name)s]: %(message)s",
            handlers=handlers,
        )


class ServeGroup(argclass.Group):
    """Web serving options."""

    listen: str = argclass.Argument(
        default="localhost:8086",
        metavar="HOST:PORT",
        help="address to listen on",
    )


class Args(argclass.Parser):
    """Axio Agent TUI."""

    serve: bool = argclass.Argument(action="store_true", help="run in web mode")  # type: ignore[call-overload]
    log = LogGroup(title="logging options")
    web = ServeGroup(title="web serving options")
