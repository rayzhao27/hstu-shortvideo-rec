"""Download and unpack a KuaiRand release into datasets/raw/.

    python -m data.download                      # KuaiRand-Pure (default)
    python -m data.download --dataset KuaiRand-1K
    python -m data.download --dataset KuaiRand-27K
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

logger = logging.getLogger("data.download")

RAW_DIR = Path("datasets/raw")

_ZENODO_FILES = "https://zenodo.org/records/10439422/files"
_USER_AGENT = "hstu-kuairand/0.1"
_CHUNK_SIZE = 1 << 20


@dataclass(frozen=True)
class DatasetSpec:
    """A downloadable KuaiRand release."""

    name: str
    url: str
    md5: str
    archive_name: str
    data_subdir: str  # data directory inside the extracted archive
    archive_mib: float
    extracted_note: str

    @classmethod
    def zenodo(cls, name: str, md5: str, archive_mib: float, extracted_note: str) -> DatasetSpec:
        archive = f"{name}.tar.gz"
        return cls(
            name=name,
            url=f"{_ZENODO_FILES}/{archive}",
            md5=md5,
            archive_name=archive,
            data_subdir=f"{name}/data",
            archive_mib=archive_mib,
            extracted_note=extracted_note,
        )


# Checksums and sizes come from the Zenodo record API for 10439422.
# Pure keeps only the logs of videos inside the candidate pool.
KUAIRAND_PURE = DatasetSpec.zenodo(
    "KuaiRand-Pure", "0820331067a3784d9691136f772b35a7", 45.2, "194MB extracted"
)
KUAIRAND_1K = DatasetSpec.zenodo(
    "KuaiRand-1K", "6b0b9c8222d67fcd4c676218edca3f1f", 1082.8, "4.3GB extracted"
)
KUAIRAND_27K = DatasetSpec.zenodo(
    "KuaiRand-27K", "3e3c799a24e2d23a4d2c757fbf9adf59", 9433.9, "46GB extracted"
)

DATASETS = {spec.name: spec for spec in (KUAIRAND_PURE, KUAIRAND_1K, KUAIRAND_27K)}


def md5sum(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _stream_download(url: str, dest: Path) -> None:
    """Download url to dest, resuming a previous partial attempt if any."""
    partial = dest.with_suffix(dest.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0

    headers = {"User-Agent": _USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        logger.info("Resuming download at %.1f MB", offset / 1024**2)

    with urlopen(Request(url, headers=headers), timeout=60) as response:
        resuming = response.status == 206
        if offset and not resuming:
            logger.warning("Server ignored the range request, restarting from scratch")
            offset = 0
        total = int(response.headers.get("Content-Length", 0)) + offset

        with open(partial, "ab" if resuming else "wb") as handle:
            done = offset
            next_report = 10
            while True:
                block = response.read(_CHUNK_SIZE)
                if not block:
                    break
                handle.write(block)
                done += len(block)
                if total and done * 100 // total >= next_report:
                    percent = done * 100 // total
                    logger.info(
                        "  %3d%%  %.1f / %.1f MB", percent, done / 1024**2, total / 1024**2
                    )
                    next_report = percent + 10

    partial.replace(dest)


def download_archive(spec: DatasetSpec, root: Path, force: bool = False) -> Path:
    """Return a md5-verified local copy of the archive, downloading if needed."""
    root.mkdir(parents=True, exist_ok=True)
    archive = root / spec.archive_name

    if archive.exists() and not force:
        if md5sum(archive) == spec.md5:
            logger.info("Archive already downloaded and verified: %s", archive)
            return archive
        logger.warning("Checksum mismatch on the existing archive, re-downloading")

    logger.info("Downloading %s (%.1f MiB, %s)", spec.url, spec.archive_mib, spec.extracted_note)
    _stream_download(spec.url, archive)

    digest = md5sum(archive)
    if digest != spec.md5:
        raise RuntimeError(
            f"md5 mismatch for {archive}: got {digest}, expected {spec.md5}. "
            "Delete the file and retry."
        )
    logger.info("md5 verified: %s", digest)
    return archive


def extract_archive(archive: Path, root: Path) -> None:
    logger.info("Extracting %s", archive.name)
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(root, filter="data")  # Python 3.12+ wants an explicit filter
        except TypeError:
            tar.extractall(root)


def ensure_dataset(spec: DatasetSpec, root: Path = RAW_DIR, download: bool = True) -> Path:
    """Return the data directory of the release under root, fetching it if absent."""
    data_dir = root / spec.data_subdir
    if data_dir.is_dir() and any(data_dir.glob("*.csv")):
        logger.info("Dataset already available: %s", data_dir)
        return data_dir

    if not download:
        raise FileNotFoundError(
            f"{data_dir} not found and downloading is disabled. "
            f"Fetch {spec.url} manually or drop --no-download."
        )

    archive = download_archive(spec, root)
    extract_archive(archive, root)

    if not data_dir.is_dir():
        raise RuntimeError(f"{spec.archive_name} did not contain {spec.data_subdir}")
    logger.info("Dataset ready: %s", data_dir)
    return data_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a KuaiRand release into datasets/raw/.")
    parser.add_argument("--dataset", default="KuaiRand-Pure", choices=sorted(DATASETS))
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--force", action="store_true", help="re-download even if verified")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)-16s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    spec = DATASETS[args.dataset]
    if args.force:
        archive = download_archive(spec, args.raw_dir, force=True)
        extract_archive(archive, args.raw_dir)
        print(args.raw_dir / spec.data_subdir)
    else:
        print(ensure_dataset(spec, args.raw_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
