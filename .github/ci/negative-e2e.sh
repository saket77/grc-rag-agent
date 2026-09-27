#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

port="${CI_NEGATIVE_PORT:-8770}"
artifacts="$(mktemp -d)"
server_pid=""
python_command="python"
if [[ -x .venv/bin/python ]]; then
  python_command=".venv/bin/python"
fi

cleanup() {
  status=$?
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  if [[ $status -ne 0 ]]; then
    printf '%s\n' "Negative-bench server log:"
    sed -n '1,240p' "$artifacts/server.log" 2>/dev/null || true
  fi
  rm -rf "$artifacts"
  return "$status"
}
trap cleanup EXIT

unset OPENAI_API_KEY OPENAI_BASE_URL LANGSMITH_API_KEY
export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false
export PYTHONUNBUFFERED=1

start_server() {
  app_target="$1"
  : >"$artifacts/server.log"
  "$python_command" -m uvicorn "$app_target" \
    --app-dir .github/ci \
    --host 127.0.0.1 \
    --port "$port" \
    --no-access-log \
    >"$artifacts/server.log" 2>&1 &
  server_pid=$!
  ready=""
  for _ in {1..60}; do
    kill -0 "$server_pid" 2>/dev/null || break
    if curl --silent --show-error --fail \
      "http://127.0.0.1:${port}/healthz" >/dev/null 2>&1; then
      ready="yes"
      break
    fi
    sleep 0.25
  done
  test "$ready" = "yes"
}

stop_server() {
  kill -INT "$server_pid"
  wait "$server_pid" || true
  server_pid=""
}

run_profile() {
  app_target="$1"
  profile="$2"
  start_server "$app_target"
  "$python_command" -m scripts.negative_http_bench \
    --url "http://127.0.0.1:${port}" \
    --profile "$profile"
  stop_server
}

run_profile "e2e_app:app" "validation"
run_profile "negative_app:provider_timeout_app" "provider-timeout"
run_profile "negative_app:request_timeout_app" "request-timeout"
run_profile "negative_app:provider_unavailable_app" "provider-unavailable"
run_profile "negative_app:invalid_evidence_app" "invalid-evidence"
run_profile "negative_app:busy_app" "busy"

printf '%s\n' "Real-HTTP negative bench passed."
