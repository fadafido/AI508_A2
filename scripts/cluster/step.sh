#!/usr/bin/env bash
# Works through data/cluster/plan.txt for at most BUDGET seconds: start the cluster, run the jobs that
# are not done yet, stop the cluster. Run it again until it says "all done". The VM I used kills
# background processes between sessions, so the campaign had to be split up like this.
# A job counts as done when progress.txt has it as ok, or as failed twice.
#
#   scripts/cluster/step.sh [budget_seconds]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

# The VM's hostname doesn't resolve and the NameNode won't start without that, so run everything in
# a private UTS namespace where the hostname is "localhost" (no root needed).
if ! getent hosts "$(hostname)" >/dev/null && [ -z "${IN_NS:-}" ]; then
  exec unshare -r -u env IN_NS=1 bash -c 'hostname localhost && exec bash "$0" "$@"' "$0" "$@"
fi
source "$HERE/env.sh"
BUDGET=${1:-150}
PLAN=${PLAN:-$OUT/plan.txt}
PROG=$OUT/progress.txt
touch "$PROG"
T0=$(date +%s)
elapsed() { echo $(( $(date +%s) - T0 )); }

pending() {
  awk -F'\t' 'FILENAME == ARGV[1] { if ($2 == "ok") ok[$1] = 1; else if ($2 == "failed") f[$1]++; next }
              NF && !($0 in ok) && f[$0] < 2' "$PROG" "$PLAN"
}

# rough duration of a job, from earlier jobs of the same kind/workload/profile if there are any
guess() {
  local t
  t=$(awk -F'\t' -v k="$1 $2 $3" 'index($1, k) == 1 && $2 == "ok" {s += $3; n++} END {if (n) printf "%d", s / n}' "$PROG")
  if [ -n "$t" ]; then echo $(( t + 5 )); elif [ "$3" = tiny ]; then echo 30; else echo 60; fi
}

n=$(pending | wc -l)
if [ "$n" -eq 0 ]; then echo "all done"; exit 0; fi

bash "$HERE/stop.sh" >/dev/null 2>&1
rm -rf "$WORK"/spark-work/app-*
bash "$HERE/start.sh" || { echo "cluster did not start"; exit 1; }
echo "cluster up after $(elapsed)s, $n jobs left"

while read -r job; do
  [ -z "$job" ] && continue
  set -- $job
  need=$(guess "$1" "$2" "$3")
  [ $(( $(elapsed) + need + 15 )) -gt "$BUDGET" ] && break
  t=$(date +%s)
  if bash "$HERE/hibench.sh" $job >/dev/null 2>"$WORK/last_job.err"; then st=ok; else st=failed; fi
  d=$(( $(date +%s) - t ))
  printf '%s\t%s\t%s\n' "$job" "$st" "$d" >> "$PROG"
  echo "$st  $job  (${d}s)"
done < <(pending)

bash "$HERE/stop.sh" >/dev/null 2>&1
echo "$(pending | wc -l) jobs left, this session took $(elapsed)s"
