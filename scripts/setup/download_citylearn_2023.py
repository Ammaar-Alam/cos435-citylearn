import argparse
import json

from cos435_citylearn.dataset import download_citylearn_2023


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="dataset directory to download from the official 2023 bundle",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="download all official 2023 dataset directories",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="concurrent download workers",
    )
    args = parser.parse_args()
    datasets = ["all"] if args.all else args.dataset
    payload = download_citylearn_2023(datasets=datasets, max_workers=args.max_workers)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
