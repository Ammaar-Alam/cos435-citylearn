import argparse

import uvicorn

from cos435_citylearn.api.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8001, type=int)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.reload:
        uvicorn.run(
            "cos435_citylearn.api.app:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
        )
    else:
        uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
