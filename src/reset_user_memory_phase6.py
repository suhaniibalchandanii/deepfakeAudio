"""Explicitly reset one user profile after confirmation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import PROJECT_ROOT
from src.memory_phase4 import UserMemory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    parser.add_argument(
        "--confirm",
        required=True,
        help="Must exactly match the supplied user ID.",
    )
    args = parser.parse_args()
    if args.confirm != args.user_id:
        raise ValueError("--confirm must exactly match --user-id")
    memory = UserMemory(PROJECT_ROOT / "memory" / "users")
    path: Path = memory.path_for(args.user_id)
    if not path.exists():
        print("No stored memory exists for this user.")
        return
    # This CLI is intentionally explicit because removal cannot be undone.
    path.unlink()
    print(f"Deleted user memory: {path}")


if __name__ == "__main__":
    main()
