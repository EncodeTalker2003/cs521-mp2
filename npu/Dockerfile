# The NKI simulator and the starter code, for machines that cannot install the
# Neuron compiler (neuronx-cc) directly, such as Macs. neuronx-cc is only
# published for Linux on x86-64, so the image is always built for linux/amd64;
# on an Apple Silicon Mac, Docker runs it under x86-64 emulation.
#
# Build, from the npu directory:
#     docker build --platform linux/amd64 -t mp2-nki-sim .
# Run the tests on your own copy of the code:
#     docker run --rm -it --platform linux/amd64 -v "$PWD":/mp2/npu mp2-nki-sim \
#         python3 test_harness.py --simulate

FROM --platform=linux/amd64 python:3.12-slim

# Install the simulator with the same script used on Linux, so the compiler
# version is pinned in one place.
COPY install_simulator.sh /tmp/install_simulator.sh
RUN NKI_ENV=/opt/nki-env bash /tmp/install_simulator.sh \
    && rm /tmp/install_simulator.sh \
    && rm -rf /root/.cache/pip

# What activating the environment does, plus unbuffered output so that every
# test result line is printed, and no __pycache__ written into a mounted copy.
ENV VIRTUAL_ENV=/opt/nki-env \
    PATH="/opt/nki-env/bin:$PATH" \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# A copy of the starter code. Mount your own copy over it to keep your edits.
COPY . /mp2/npu
WORKDIR /mp2/npu
CMD ["bash"]
