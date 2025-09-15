#!/usr/bin/env python3
"""
Simple script: Get the git commit hash of LLaMA Factory
"""

import subprocess
import sys


def main():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd="/llamafactory", capture_output=True, text=True, check=True
        )
        print(result.stdout.strip())
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
