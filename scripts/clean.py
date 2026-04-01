import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Python cache artifacts.")
    parser.add_argument(
        "--root",
        default=".",
        help="Repo root to clean (default: .)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    removed_dirs = 0
    removed_files = 0

    for path in root.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed_dirs += 1

    for pattern in ("*.pyc", "*.pyo"):
        for path in root.rglob(pattern):
            if path.is_file():
                try:
                    path.unlink()
                    removed_files += 1
                except FileNotFoundError:
                    pass

    print(f"Removed __pycache__ dirs: {removed_dirs}")
    print(f"Removed bytecode files:  {removed_files}")


if __name__ == "__main__":
    main()

