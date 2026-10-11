#!/usr/bin/env bash
# Run an explicit experiment config in a CPU pod and retrieve its complete artifacts.
set -euo pipefail

cluster_dir="${1:?Usage: $0 <cluster-dir> <crossover|bandwidth> <config.json> [options]}"
command="${2:?experiment command required}"
config="${3:?experiment config required}"
shift 3
case "$command" in crossover|bandwidth) ;; *) echo "Unsupported experiment: $command" >&2; exit 2 ;; esac
root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ ! -f "$cluster_dir/env.sh" || ! -f "$config" || ! -f "$root/toolkit/$command.py" ]]; then
    echo "Cluster env, experiment config, and runner must exist" >&2
    exit 2
fi
source "$cluster_dir/env.sh"
python3 "$root/toolkit/$command.py" validate "$config"

pod="${TEST_CLIENT:-test-client}"
run_id="$command-$(date -u +%Y%m%dT%H%M%SZ)-$(od -An -N6 -tx1 /dev/urandom | tr -d ' \n')"
remote_root="${MEASUREMENT_REMOTE_ROOT:-/tmp/llm-d-diagnostics}"
if [[ "$remote_root" != /* ]]; then
    echo "MEASUREMENT_REMOTE_ROOT must be an absolute pod path" >&2
    exit 2
fi
remote="$remote_root/$run_id"
destination="${DATA_DIR:?DATA_DIR is required}/$run_id"
mkdir -p "$(dirname "$destination")"
mkdir "$destination"
oc exec "$pod" -n "${NS:?NS is required}" -- mkdir -p "$remote"
for file in experiment_io.py "$command.py"; do
    oc cp "$root/toolkit/$file" "$NS/$pod:$remote/$file"
done
oc cp "$config" "$NS/$pod:$remote/config.json"
commit="$(git -C "$root" rev-parse HEAD 2>/dev/null || echo unknown)"
if [[ -n "$(git -C "$root" status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
    commit="$commit-dirty"
fi
remote_env=("TOOLKIT_COMMIT=$commit" "PYTHONDONTWRITEBYTECODE=1")
if [[ -n "${EXPERIMENT_API_KEY:-}" ]]; then
    remote_env+=("EXPERIMENT_API_KEY=$EXPERIMENT_API_KEY")
fi
status=0
oc exec "$pod" -n "$NS" -- env "${remote_env[@]}" python3 "$remote/$command.py" \
    run "$remote/config.json" --output "$remote/results" "$@" || status=$?

if ! oc exec "$pod" -n "$NS" -- tar -C "$remote" -czf - results \
        | tar -C "$destination" -xzf -; then
    echo "Result copy failed; remote artifacts retained at $NS/$pod:$remote" >&2
    exit 1
fi
PYTHONPATH="$root/toolkit" python3 - "$destination/results" <<'PY'
import sys
from experiment_io import verify_manifest
verify_manifest(sys.argv[1])
PY
oc exec "$pod" -n "$NS" -- rm -rf "$remote"
echo "Artifacts: $destination/results (experiment exit status: $status)"
exit "$status"
