"""``python -m mlx_dspark`` entry point — delegates to the subcommand router in cli.py.

  python -m mlx_dspark serve --model mlx-community/Qwen3-8B-8bit   # OpenAI-compatible API server
  python -m mlx_dspark generate --model mlx-community/Qwen3-4B-8bit --prompt "Explain rainbows."
  python -m mlx_dspark models                          # targets with an auto-resolved drafter
  python -m mlx_dspark doctor                           # environment check

The historical flat form (no subcommand) still works and maps to ``generate``:

  python -m mlx_dspark --mode baseline --prompt "..." --max-new-tokens 220
  python -m mlx_dspark --mode dspark   --prompt "..." --max-new-tokens 220
"""

from .cli import main

if __name__ == "__main__":
    main()
