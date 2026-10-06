#!/bin/bash
#
# Installs the NKI simulator, which is part of the Neuron compiler (neuronx-cc),
# so you can develop and test conv_npu.py on your own machine before you have
# Trainium access. The simulator checks correctness only; it does not measure
# performance.
#
# Usage, from the npu directory:
#
#     bash install_simulator.sh
#     source ~/nki-env/bin/activate
#     python3 test_harness.py --simulate
#
# On Google Colab, run `!bash install_simulator.sh` instead. It installs into
# the notebook's Python, so there is no environment to activate; run
# `%env NEURON_PLATFORM_TARGET_OVERRIDE=trn2` in the notebook instead.
#
# Optional environment variables:
#     NKI_ENV  where to create the virtual environment (default: ~/nki-env)
#     PYTHON   the Python 3.10-3.12 interpreter to use (default: auto-detect)

set -euo pipefail

# neuronx-cc 2.22 comes from Neuron SDK 2.27, which still supports the
# neuronxcc.nki API that the starter code uses.
NEURONX_CC_VERSION="2.22.*"
# The chip the simulator models: the course's trn2.3xlarge instances.
SIMULATION_TARGET="trn2"
NEURON_INDEX="https://pip.repos.neuron.amazonaws.com"
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
NKI_ENV="${NKI_ENV:-$HOME/nki-env}"

die() {
    echo "error: $*" >&2
    exit 1
}

supported_python() {
    "$1" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 12))' 2>/dev/null
}

# neuronx-cc is only published for Linux on x86-64.
os=$(uname -s)
arch=$(uname -m)
if [ "$os" != "Linux" ] || [ "$arch" != "x86_64" ]; then
    die "neuronx-cc is only published for Linux on x86-64, and this machine is $os on $arch. On macOS, use Google Colab or a Linux machine. On Windows, use WSL2."
fi

if [ -n "${PYTHON:-}" ]; then
    supported_python "$PYTHON" || die "PYTHON=$PYTHON is not Python 3.10, 3.11, or 3.12."
else
    for candidate in python3 python3.12 python3.11 python3.10; do
        if command -v "$candidate" > /dev/null && supported_python "$candidate"; then
            PYTHON=$candidate
            break
        fi
    done
    [ -n "${PYTHON:-}" ] || die "neuronx-cc $NEURONX_CC_VERSION needs Python 3.10, 3.11, or 3.12, and none was found. Install one, or point PYTHON at one."
fi

if "$PYTHON" -c 'import google.colab' 2> /dev/null; then
    on_colab=1
    echo "Google Colab detected: installing into the notebook's Python."
else
    on_colab=0
    if [ ! -e "$NKI_ENV" ]; then
        echo "Creating a virtual environment in $NKI_ENV"
        if ! "$PYTHON" -m venv "$NKI_ENV"; then
            # Don't leave a half-built environment behind for the next run.
            rm -rf "$NKI_ENV"
            die "could not create the virtual environment (see the message above)."
        fi
    fi
    supported_python "$NKI_ENV/bin/python" \
        || die "$NKI_ENV exists but is not a Python 3.10-3.12 virtual environment. Delete it, or set NKI_ENV to another path."
    PYTHON="$NKI_ENV/bin/python"

    # Activating the environment also selects the chip the simulator models.
    if ! grep -q "NEURON_PLATFORM_TARGET_OVERRIDE" "$NKI_ENV/bin/activate"; then
        printf '\n# Simulate the trn2 chip in the course'"'"'s Trainium instances.\nexport NEURON_PLATFORM_TARGET_OVERRIDE=%s\n' \
            "$SIMULATION_TARGET" >> "$NKI_ENV/bin/activate"
    fi
fi

echo "Installing neuronx-cc $NEURONX_CC_VERSION"
"$PYTHON" -m pip install "neuronx-cc==$NEURONX_CC_VERSION" --extra-index-url "$NEURON_INDEX"

# The test harness uses PyTorch to compute the reference output.
if ! "$PYTHON" -c 'import torch' 2> /dev/null; then
    echo "Installing PyTorch (CPU build)"
    "$PYTHON" -m pip install torch --index-url "$TORCH_INDEX"
fi

echo "Checking that the simulator runs"
# nki.jit reads the kernel's source code, so the check must live in a file.
check_script=$(mktemp --suffix=.py)
trap 'rm -f "$check_script"' EXIT
cat > "$check_script" << 'EOF'
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit
def add_one(a):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.hbm)
    nl.store(out, value=nl.add(nl.load(a), 1.0))
    return out


a = np.arange(8, dtype=np.float32).reshape(2, 4)
assert np.array_equal(nki.simulate_kernel(add_one, a), a + 1)
EOF
NEURON_PLATFORM_TARGET_OVERRIDE="$SIMULATION_TARGET" "$PYTHON" "$check_script" > /dev/null

echo
echo "The NKI simulator is installed."
if [ "$on_colab" = 1 ]; then
    echo "In the notebook, run:"
    echo "    %env NEURON_PLATFORM_TARGET_OVERRIDE=$SIMULATION_TARGET"
    echo "Then, from the npu directory, run:"
    echo "    !python3 test_harness.py --simulate"
else
    echo "Activate the environment in every new shell with:"
    echo "    source $NKI_ENV/bin/activate"
    echo "Then, from the npu directory, run:"
    echo "    python3 test_harness.py --simulate"
fi
