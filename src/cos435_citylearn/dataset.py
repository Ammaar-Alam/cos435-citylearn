from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from cos435_citylearn.io import write_json
from cos435_citylearn.paths import DATA_DIR
from cos435_citylearn.runtime import utc_now_iso

CITYLEARN_2023_DOI = "doi:10.18738/T8/SXFWTI"
CITYLEARN_2023_PERSISTENT_URL = "https://doi.org/10.18738/T8/SXFWTI"
CITYLEARN_2023_API_URL = (
    "https://dataverse.tdl.org/api/datasets/:persistentId/?persistentId=doi:10.18738/T8/SXFWTI"
)
CITYLEARN_2023_DATAFILE_URL = "https://dataverse.tdl.org/api/access/datafile"
DEFAULT_DATASET_NAME = "citylearn_challenge_2023_phase_2_local_evaluation"
DATASET_ROOT = DATA_DIR / "external" / "citylearn_2023"
MANIFEST_PATH = DATA_DIR / "manifests" / "citylearn_2023_manifest.json"
USER_AGENT = "cos435-citylearn/0.1"
DEFAULT_TIMEOUT = 300


@dataclass(frozen=True)
class RemoteFile:
    directory_label: str
    datafile_id: int
    filename: str
    md5: str
    size_bytes: int
    source_label: str

    @property
    def relative_path(self) -> Path:
        return Path(self.directory_label) / self.filename


def _request(url: str):
    return Request(url, headers={"User-Agent": USER_AGENT})


def fetch_dataset_metadata() -> dict[str, Any]:
    with urlopen(_request(CITYLEARN_2023_API_URL), timeout=DEFAULT_TIMEOUT) as response:
        payload = json.load(response)

    return payload["data"]["latestVersion"]


def available_dataset_names(metadata: dict[str, Any] | None = None) -> list[str]:
    version = fetch_dataset_metadata() if metadata is None else metadata
    names = {entry["directoryLabel"] for entry in version["files"]}
    return sorted(names)


def select_dataset_names(
    requested: list[str] | None,
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    version = fetch_dataset_metadata() if metadata is None else metadata
    available = available_dataset_names(version)

    if not requested:
        return [DEFAULT_DATASET_NAME]

    if requested == ["all"]:
        return available

    missing = sorted(set(requested) - set(available))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"unknown CityLearn 2023 dataset names: {joined}")

    return requested


def build_remote_files(
    metadata: dict[str, Any],
    selected_datasets: list[str],
) -> list[RemoteFile]:
    files = []

    for entry in metadata["files"]:
        directory_label = entry["directoryLabel"]
        if directory_label not in selected_datasets:
            continue

        data_file = entry["dataFile"]
        filename = data_file.get("originalFileName") or data_file["filename"]
        files.append(
            RemoteFile(
                directory_label=directory_label,
                datafile_id=int(data_file["id"]),
                filename=filename,
                md5=data_file["checksum"]["value"],
                size_bytes=int(data_file["filesize"]),
                source_label=entry["label"],
            )
        )

    return sorted(files, key=lambda item: item.relative_path.as_posix())


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(remote_file: RemoteFile, destination_root: Path) -> dict[str, Any]:
    destination = destination_root / remote_file.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and _md5_file(destination) == remote_file.md5:
        status = "cached"
    else:
        temp_path = destination.with_suffix(destination.suffix + ".part")
        url = f"{CITYLEARN_2023_DATAFILE_URL}/{remote_file.datafile_id}?format=original"
        with urlopen(_request(url), timeout=DEFAULT_TIMEOUT) as response:
            with temp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)

        actual_md5 = _md5_file(temp_path)
        if actual_md5 != remote_file.md5:
            temp_path.unlink(missing_ok=True)
            raise ValueError(
                f"checksum mismatch for {remote_file.relative_path}: "
                f"expected {remote_file.md5}, got {actual_md5}"
            )

        temp_path.replace(destination)
        status = "downloaded"

    return {
        "datafile_id": remote_file.datafile_id,
        "directory_label": remote_file.directory_label,
        "filename": remote_file.filename,
        "relative_path": remote_file.relative_path.as_posix(),
        "md5": remote_file.md5,
        "size_bytes": remote_file.size_bytes,
        "source_label": remote_file.source_label,
        "status": status,
    }


def download_citylearn_2023(
    datasets: list[str] | None = None,
    destination_root: str | Path | None = None,
    max_workers: int = 4,
) -> dict[str, Any]:
    destination = DATASET_ROOT if destination_root is None else Path(destination_root)
    destination.mkdir(parents=True, exist_ok=True)

    metadata = fetch_dataset_metadata()
    selected = select_dataset_names(datasets, metadata)
    remote_files = build_remote_files(metadata, selected)
    worker_count = max(1, min(max_workers, len(remote_files)))

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        file_entries = list(pool.map(lambda item: _download_file(item, destination), remote_files))

    file_entries = sorted(file_entries, key=lambda entry: entry["relative_path"])
    manifest = {
        "generated_at": utc_now_iso(),
        "dataset_doi": CITYLEARN_2023_DOI,
        "persistent_url": CITYLEARN_2023_PERSISTENT_URL,
        "api_url": CITYLEARN_2023_API_URL,
        "download_root": str(destination),
        "downloaded_datasets": selected,
        "default_dataset": DEFAULT_DATASET_NAME,
        "all_available_datasets": available_dataset_names(metadata),
        "publication_date": metadata["publicationDate"],
        "version_number": metadata["versionNumber"],
        "version_minor_number": metadata["versionMinorNumber"],
        "downloaded_file_count": len(file_entries),
        "files": file_entries,
    }
    write_json(MANIFEST_PATH, manifest)
    return manifest
