"""Build/update the local MLB Big Data warehouse from repository datasets."""

import json
from modules.bigdata_mlb import bootstrap_from_repository


if __name__ == "__main__":
    result = bootstrap_from_repository()
    print(json.dumps(result, indent=2, default=str))
