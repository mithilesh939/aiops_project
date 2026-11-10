"""Roll the production model back to an earlier registry version.

Usage:
    python scripts/rollback.py            # go back to the previous version
    python scripts/rollback.py --to 1     # go to a specific version

Moves the `production` alias in the MLflow registry, then asks the running
API to reload so the change takes effect without a restart.
"""
import argparse
import os
import urllib.request

from mlflow.tracking import MlflowClient

MODEL = "quizpulse-kt"
ALIAS = "production"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", type=int, help="target version (default: previous)")
    ap.add_argument("--api", default=os.environ.get("API_URL", "http://localhost:8000"))
    args = ap.parse_args()

    client = MlflowClient(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    current = int(client.get_model_version_by_alias(MODEL, ALIAS).version)
    if args.to is not None:
        target = args.to
    else:
        older = sorted(int(v.version) for v in client.search_model_versions(f"name='{MODEL}'")
                       if int(v.version) < current)
        if not older:
            raise SystemExit(f"v{current} is the oldest version; nothing to roll back to")
        target = older[-1]

    client.set_registered_model_alias(MODEL, ALIAS, str(target))
    client.set_model_version_tag(MODEL, str(target), "rolled_back_from", str(current))
    print(f"alias '{ALIAS}': v{current} -> v{target}")
    try:
        req = urllib.request.Request(f"{args.api}/admin/reload", method="POST")
        print("API reload:", urllib.request.urlopen(req, timeout=30).read().decode())
    except Exception as exc:  # API may be down; alias is already moved
        print(f"alias moved, but API reload failed ({exc}); the poller will pick it up")


if __name__ == "__main__":
    main()
