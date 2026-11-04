#!/usr/bin/env bash
# Run from the repository root. The wrapper owns durable launcher logs/status.
set -euo pipefail
if (( $# < 3 )); then
    echo 'Usage: scripts/launch.sh GPU RUN_NAME dlzo-command [arguments...]' >&2
    exit 2
fi
gpu="$1"; name="$2"; shift 2
[[ "$gpu" =~ ^[0-9]+$ && "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
if [[ "$1" == "es-countdown" ]]; then
    shift
    command=(python -m extensions.es_countdown.train "$@")
else
    command=(dlzo "$@")
fi
command -v "${command[0]}" >/dev/null
command -v setsid >/dev/null
IFS=',' read -r memory utilization < <(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
(( memory < 1024 && utilization < 10 )) || { echo "GPU $gpu is occupied" >&2; exit 1; }
mkdir -p runs/launches
folder="runs/launches/$name"
mkdir "$folder"
printf '%s\n' "$gpu" > "$folder/gpu.txt"
printf '%q ' "${command[@]}" > "$folder/command.txt"
printf '\n' >> "$folder/command.txt"
nohup setsid env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    bash -c '
        folder="$1"; shift
        printf "running\n" > "$folder/status.txt"
        "$@"
        code=$?
        printf "exited\t%s\n" "$code" > "$folder/status.txt"
        exit "$code"
    ' _ "$folder" "${command[@]}" > "$folder/console.log" 2>&1 < /dev/null &
echo "$!" > "$folder/pid.txt"
echo "PID $!; GPU $gpu; log $folder/console.log"
# Exit status alone does not establish completion. Inspect run.json and metric artifacts.
