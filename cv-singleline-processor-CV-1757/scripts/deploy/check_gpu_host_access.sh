#!/usr/bin/env bash
# Quick connectivity check for the RTX 5090 host(s).
# Run from your Mac while connected to the LF/VPN network.
set -euo pipefail

KEY="${HOME_DEPOT_SSH_KEY:-$HOME/Downloads/HomeDepotCV/avinash_patel_lf.pem}"
USER="${HOME_DEPOT_SSH_USER:-avinash.patel}"
HOSTS=(
  "172.16.20.108"  # GPU5-A5090 (documented 5090 machine)
  "172.16.20.104"  # alternate .104 host if team uses this
)

echo "Using key: $KEY"
echo "Using user: $USER"
echo

for host in "${HOSTS[@]}"; do
  echo "=== $host ==="
  if ping -c 1 -W 2 "$host" >/dev/null 2>&1; then
    echo "  ping: OK"
  else
    echo "  ping: no reply (ICMP may be blocked; trying SSH anyway)"
  fi

  if nc -z -G 5 "$host" 22 2>/dev/null; then
    echo "  SSH port 22: open"
  else
    echo "  SSH port 22: closed or unreachable"
    echo
    continue
  fi

  if [[ -f "$KEY" ]]; then
    if ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
      "${USER}@${host}" 'hostname; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -2; ls -d ~/HomeDepotCV 2>/dev/null || echo "HomeDepotCV: not found"' 2>/dev/null; then
      echo "  SSH login: OK"
    else
      echo "  SSH login: failed (wrong key/user or auth disabled)"
    fi
  else
    echo "  SSH key not found at $KEY"
  fi
  echo
done
