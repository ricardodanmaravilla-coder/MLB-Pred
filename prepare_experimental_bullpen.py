"""Build deeper bullpen history for the isolated Parquet experiment only."""
import build_bullpen_usage_history as builder

# 2023-2026 provides majority coverage of the 2021-2026 research corpus while
# keeping the official-game-feed collection bounded for CI.
builder.START_SEASON = 2023
builder.MAX_WORKERS = 12

if __name__ == "__main__":
    builder.main()
