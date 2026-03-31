import shlex
import sys

from textual_serve.server import Server

from .app import AgentApp


def main() -> None:
    from .args import Args

    args = Args().parse_args()
    args.log.configure()

    if args.serve:
        host, _, port_str = args.web.listen.partition(":")
        port = int(port_str) if port_str else 8086
        host = host or "localhost"
        server = Server(
            command=shlex.join([sys.executable, "-m", "axio_tui"]),
            host=host,
            port=port,
            title="Axio Agent",
        )
        server.serve()
    else:
        app = AgentApp()
        app.run()


if __name__ == "__main__":
    main()
