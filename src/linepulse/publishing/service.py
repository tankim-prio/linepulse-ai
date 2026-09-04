"""Publish validated staged CSVs using immutable content-addressed files."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path


_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class PublishResult:
    dataset_name: str
    content_sha256: str
    published_path: str
    pointer_path: str
    already_published: bool


class DatasetPublisher:
    """Atomically expose a validated staged dataset."""

    def publish(
        self,
        dataset_name: str,
        staged_path: Path | str,
        publish_root: Path | str,
    ) -> PublishResult:

        if not _SAFE_NAME.fullmatch(dataset_name):
            raise ValueError("Unsafe dataset name.")

        source = Path(staged_path).resolve()

        if not source.is_file():
            raise FileNotFoundError(
                f"Staged file does not exist: {source}"
            )

        if source.suffix.lower() != ".csv":
            raise ValueError("Published source must be a CSV file.")

        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()

        expected_name = f"{dataset_name}-{digest}.csv"

        if source.name != expected_name:
            raise ValueError(
                "Staged filename does not match dataset name "
                "and SHA-256 content."
            )

        root = Path(publish_root).resolve()
        root.mkdir(parents=True, exist_ok=True)

        dataset_dir = (root / dataset_name).resolve()

        if dataset_dir.parent != root:
            raise ValueError(
                "Dataset publish directory escaped publish root."
            )

        dataset_dir.mkdir(parents=True, exist_ok=True)

        destination = (
            dataset_dir / expected_name
        ).resolve()

        if destination.parent != dataset_dir:
            raise ValueError(
                "Published file escaped dataset directory."
            )

        created = False

        if not destination.exists():
            try:
                with destination.open("xb") as handle:
                    handle.write(payload)
                created = True
            except FileExistsError:
                created = False

        # Verify existing immutable content.
        published_digest = hashlib.sha256(
            destination.read_bytes()
        ).hexdigest()

        if published_digest != digest:
            raise RuntimeError(
                "Published immutable file has unexpected content."
            )

        pointer = dataset_dir / "CURRENT"

        previous = None
        if pointer.exists():
            previous = pointer.read_text(
                encoding="utf-8"
            ).strip()

        already_published = (
            previous == destination.name
            and not created
        )

        if previous != destination.name:
            self._write_pointer_atomic(
                pointer,
                destination.name,
            )

        return PublishResult(
            dataset_name=dataset_name,
            content_sha256=digest,
            published_path=str(destination),
            pointer_path=str(pointer),
            already_published=already_published,
        )

    @staticmethod
    def _write_pointer_atomic(
        pointer: Path,
        filename: str,
    ) -> None:
        """Replace CURRENT atomically within the same directory."""

        temp_name = None

        try:
            fd, temp_name = tempfile.mkstemp(
                prefix=".CURRENT-",
                suffix=".tmp",
                dir=pointer.parent,
                text=True,
            )

            with os.fdopen(
                fd,
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                handle.write(filename + "\n")
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temp_name, pointer)

        finally:
            if temp_name:
                temp_path = Path(temp_name)
                if temp_path.exists():
                    temp_path.unlink()
