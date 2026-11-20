#!/usr/bin/env bash
# Runs ON the rented Vast instance: destroys the instance itself after N seconds,
# so the hard cap holds even if the controlling Mac or session disappears.
# Uses the per-instance CONTAINER_ID / CONTAINER_API_KEY that Vast injects
# (docs.vast.ai, "Using the CLI from Inside an Instance"); no account key is copied.
#
#   step5_self_destruct.sh SECONDS        verify the key, then arm the timer
#   SELF_DESTRUCT_ENVIRON=file            override /proc/1/environ (tests)
#   VASTAI=path                           override the vastai binary (tests)
set -u

seconds=${1:?usage: step5_self_destruct.sh SECONDS}
environ=${SELF_DESTRUCT_ENVIRON:-/proc/1/environ}
log=${SELF_DESTRUCT_LOG:-/root/self_destruct.log}

# ssh sessions do not inherit the container's env; fall back to PID 1's.
lookup() {
  local value="${!1:-}"
  if [ -z "$value" ] && [ -r "$environ" ]; then
    value=$(tr '\0' '\n' < "$environ" | sed -n "s/^$1=//p" | head -1)
  fi
  printf '%s' "$value"
}

id=$(lookup CONTAINER_ID)
key=$(lookup CONTAINER_API_KEY)
if [ -z "$id" ] || [ -z "$key" ]; then
  echo "SELF_DESTRUCT_UNAVAILABLE: CONTAINER_ID/CONTAINER_API_KEY not found"
  exit 1
fi

vastai=${VASTAI:-$(command -v vastai || true)}
if [ -z "$vastai" ]; then
  echo "SELF_DESTRUCT_UNAVAILABLE: vastai CLI not installed"
  exit 1
fi

# Prove the key works with a read-only call before relying on it.
if ! "$vastai" show instance "$id" --api-key "$key" --raw >/dev/null 2>&1; then
  echo "SELF_DESTRUCT_KEY_REJECTED: read-only check with the per-instance key failed"
  exit 1
fi

echo "$(date -u +%FT%TZ) armed: destroy instance $id in ${seconds}s" >> "$log"
(
  sleep "$seconds"
  while true; do
    echo "$(date -u +%FT%TZ) self-destruct: destroying instance $id" >> "$log"
    "$vastai" destroy instance "$id" --api-key "$key" -y >> "$log" 2>&1
    sleep 60
  done
) < /dev/null > /dev/null 2>&1 &
disown
echo "SELF_DESTRUCT_ARMED instance=$id seconds=$seconds pid=$!"
