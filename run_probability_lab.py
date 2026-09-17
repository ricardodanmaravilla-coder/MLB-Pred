"""Standalone Probability Lab runner.

Runs the experiment without invoking V7 production persistence.
"""
import json

from modules.enriched_web_service import EnrichedMLBWebService
from modules.probability_lab import scan_probability_lab


def main():
    service = EnrichedMLBWebService()
    result = scan_probability_lab(service, persist=True)
    print(json.dumps({
        "version": result["version"],
        "worksheet": result["worksheet"],
        "rule": result["rule"],
        "stake_rule": result["stake_rule"],
        "accepted_count": len(result["accepted"]),
        "error_count": len(result["errors"]),
        "sheet_status": result["sheet_status"],
        "accepted": result["accepted"],
        "errors": result["errors"],
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
