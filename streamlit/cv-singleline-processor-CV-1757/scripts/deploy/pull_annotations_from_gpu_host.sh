#!/usr/bin/env bash
# Pull team annotations from the 5090 host into your local golden_sku_truth.csv.
#
# Peer saves on http://172.16.20.108:8503 go to the REMOTE copy:
#   ~/HomeDepotCV/research_outputs/golden_dataset_local_tests/golden_sku_truth.csv
# They do NOT automatically appear on your Mac — run this script to fetch them.
#
# Usage:
#   ./scripts/deploy/pull_annotations_from_gpu_host.sh
#   ./scripts/deploy/pull_annotations_from_gpu_host.sh --merge   # keep local + remote labels
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
HOST="172.16.20.108"
USER="${HOME_DEPOT_SSH_USER:-avinash.patel}"
KEY="${HOME_DEPOT_SSH_KEY:-$PROJECT_ROOT/avinash_patel_lf.pem}"
MERGE=0
REMOTE_CSV="~/HomeDepotCV/research_outputs/golden_dataset_local_tests/golden_sku_truth.csv"
LOCAL_CSV="$PROJECT_ROOT/research_outputs/golden_dataset_local_tests/golden_sku_truth.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --user) USER="$2"; shift 2 ;;
    --key) KEY="$2"; shift 2 ;;
    --merge) MERGE=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

RSYNC=(rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no")
TMP_REMOTE="$(mktemp /tmp/golden_sku_truth_remote.XXXXXX.csv)"
trap 'rm -f "$TMP_REMOTE"' EXIT

echo "Pulling annotations from ${USER}@${HOST}..."
"${RSYNC[@]}" "${USER}@${HOST}:${REMOTE_CSV}" "$TMP_REMOTE"

if [[ "$MERGE" -eq 0 ]]; then
  if [[ -f "$LOCAL_CSV" ]]; then
    backup="${LOCAL_CSV%.csv}.backup_$(date +%Y%m%d_%H%M%S).csv"
    cp "$LOCAL_CSV" "$backup"
    echo "Backed up local CSV to $backup"
  fi
  cp "$TMP_REMOTE" "$LOCAL_CSV"
  echo "Wrote $LOCAL_CSV"
else
  python3 - "$LOCAL_CSV" "$TMP_REMOTE" <<'PY'
import csv, sys
from datetime import datetime
from pathlib import Path

local_path, remote_path = Path(sys.argv[1]), Path(sys.argv[2])

def load(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows, list(rows[0].keys()) if rows else []

local_rows, fields = load(local_path)
remote_rows, _ = load(remote_path)
by_key = {r.get("region_key", ""): r for r in local_rows if r.get("region_key")}

for row in remote_rows:
    key = row.get("region_key", "")
    if not key:
        continue
    remote_sku = str(row.get("expected_sku", "") or "").strip()
    if not remote_sku:
        continue
    if key not in by_key:
        by_key[key] = row
        local_rows.append(row)
        continue
    local_sku = str(by_key[key].get("expected_sku", "") or "").strip()
    if not local_sku and remote_sku:
        by_key[key].update({
            "expected_sku": remote_sku,
            "review_status": row.get("review_status", ""),
            "reviewer": row.get("reviewer", ""),
            "notes": row.get("notes", ""),
        })

backup = local_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
backup.write_bytes(local_path.read_bytes())
with local_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(local_rows)
print(f"Merged remote labels into {local_path}")
print(f"Backup: {backup}")
PY
fi

python3 - "$LOCAL_CSV" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
labeled = sum(1 for r in rows if (r.get("expected_sku") or "").strip())
print(f"Local truth CSV: {len(rows)} rows, {labeled} with expected_sku")
PY
