#!/bin/bash
#
# Runs test_harness.py on a Trainium node of the course cluster. Submit it with
# sbatch from the npu directory on the login node:
#
#     sbatch run_trainium.sh                  # correctness and performance tests
#     sbatch run_trainium.sh --profile conv   # also profile the kernel
#     squeue --me                             # check on your job
#
# Arguments after the script name are passed on to test_harness.py. The output
# goes to mp2-<job id>.out in this directory.
#
# With --profile <name>, the harness captures a profile of each performance
# test after measuring it, so profiling does not change the performance
# numbers. This script then prints the MFU of each capture and converts it to
# <name>_float32.pftrace and <name>_float16.pftrace, timelines you can open at
# https://ui.perfetto.dev after copying them to your own machine. The
# conversion needs neuron-profile, which is only installed on the Trainium
# nodes.
#
# The job takes a whole Trainium node (--exclusive) so that no other job runs
# on it at the same time and skews your performance numbers. The cluster runs
# one job per student at a time, for at most 15 minutes; a second job you
# submit waits in the queue until the first one finishes.
#
#SBATCH --job-name=mp2-conv
#SBATCH --partition=nki
#SBATCH --constraint=neuron
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=mp2-%j.out

set -euo pipefail

if [ ! -f test_harness.py ]; then
    echo "error: run sbatch from the npu directory, where test_harness.py is." >&2
    exit 1
fi

# The profile name, if any, as test_harness.py will read it from the arguments.
profile=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --profile) profile="${args[i + 1]:-}" ;;
        --profile=*) profile="${args[i]#--profile=}" ;;
    esac
done

# The Neuron compiler and runtime are installed in a shared environment on the
# Trainium nodes. The tools the harness runs, neuron-bench for the performance
# tests and neuron-profile for --profile, are in /opt/aws/neuron/bin.
source /opt/aws_neuronx_venv_pytorch/bin/activate
export PATH="/opt/aws/neuron/bin:$PATH"

cc_version=$(python3 -c 'import importlib.metadata as m; print(m.version("neuronx-cc"))' 2> /dev/null || echo "(not found)")
echo "Running on $(hostname) with neuronx-cc $cc_version"
python3 test_harness.py "$@"

if [ -n "$profile" ]; then
    echo
    for dtype in float32 float16; do
        name="${profile}_${dtype}"
        if [ ! -f "$name.neff" ] || [ ! -f "$name.ntff" ]; then
            continue
        fi
        # neuron-profile reports MFU as a fraction of peak; print a percentage.
        summary=$(neuron-profile view -n "$name.neff" -s "$name.ntff" --output-format summary-text 2> /dev/null) || true
        mfu=$(awk '$1 == "mfu_estimated_percent" { printf "%.1f%%", $2 * 100 }' <<< "$summary")
        echo "MFU ($dtype): ${mfu:-not found in the profile}"
        if neuron-profile view -n "$name.neff" -s "$name.ntff" --output-format perfetto --output-file "$name.pftrace" > /dev/null 2>&1; then
            echo "Wrote $name.pftrace"
        else
            echo "warning: could not convert $name.ntff to $name.pftrace" >&2
        fi
    done
fi
