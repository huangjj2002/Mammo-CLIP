"""
Compatibility entry point for EDL training.

Prefer editing and running run_edl_finetune.py as the primary EDL entrypoint.
This file is kept so existing commands continue to work unchanged.
"""

from run_edl_finetune import main


if __name__ == "__main__":
    main()
