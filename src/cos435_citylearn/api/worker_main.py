from __future__ import annotations

import argparse
import importlib
import json
import traceback
from pathlib import Path

from cos435_citylearn.io import write_json


def _load_callable(path: str):
    module_name, symbol_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-file", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.job_file).read_text())
    result_path = Path(request["result_path"])
    error_path = Path(request["error_path"])

    try:
        runner = _load_callable(request["callable_path"])
        payload = runner(
            config_path=request["config_path"],
            eval_config_path=request["eval_config_path"],
        )
        write_json(result_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
    except Exception as exc:
        error_payload = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(error_path, error_payload)
        print(error_payload["traceback"], flush=True)
        raise


if __name__ == "__main__":
    main()
