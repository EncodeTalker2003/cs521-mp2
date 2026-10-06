# Part 3: Optimizing convolution 2D on Neural Network Accelerators #

## License & Copyright ##

The `npu` part is modified from https://github.com/stanford-cs149/asst4-trainium.

## Overview ##

In this assignment, you will learn how to implement a convolution layer for the [AWS Trainium](https://aws.amazon.com/ai/machine-learning/trainium/) architecture, which features multiple tensor-oriented accelerated processing engines as well as software-managed on-chip storage that provides these engines high-bandwidth access to data. 

Overall, this assignment will:

1) Give you experience with the low-level details of tensor processing and managing on-chip SRAM on the accelerator.

2) Show the value of key locality-preserving optimizations like loop blocking and loop fusion.

## Environment Setup ##

You will work in two places. On your own machine, you develop your kernel and check that it is correct with the NKI simulator, as described in the next sections. On the course's Trainium cluster, AWS `trn2.3xlarge` instances that each have a Trainium2 accelerator, you run the performance tests and profile your kernel, as described in [Running on the Trainium cluster](#running-on-the-trainium-cluster).

### Installing the simulator

The NKI simulator runs your kernel on an ordinary CPU and checks its output, so you can get a correct kernel working before you have Trainium access. It does not measure performance: the performance tests and `neuron-profile` need the Trainium instance.

The simulator is part of the Neuron compiler package, `neuronx-cc`, which is only published for Linux on x86-64. The install script uses version 2.22, from Neuron SDK 2.27, which needs Python 3.10, 3.11, or 3.12. On Windows, use WSL2. On macOS, use the [Docker image](#macos-running-the-simulator-in-docker) described below, Google Colab, or a Linux machine.

Download the starter code and **run the simulator install script we have provided** from the `npu` directory:
```
git clone https://github.com/CS521-ml-compilers/mp2.git
cd mp2/npu
bash install_simulator.sh
```
The script creates a Python [virtual environment](https://builtin.com/data-science/python-virtual-environment) in `~/nki-env` (set `NKI_ENV` to put it somewhere else), installs the Neuron compiler and a CPU build of PyTorch, which the test harness uses for its reference output, and then checks that the simulator runs. Activate the environment in every new shell:
```
source ~/nki-env/bin/activate
```
Activating it also sets `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, which makes the simulator apply the hardware rules of the Trainium2 chip in the course's instances.

On Google Colab, run `!bash install_simulator.sh` from the `npu` directory instead. It installs into the notebook's Python, so there is nothing to activate, but run `%env NEURON_PLATFORM_TARGET_OVERRIDE=trn2` in the notebook to simulate the right chip. In a notebook, put a `!` in front of shell commands and use `%cd` to change directories.

Do not run `install.sh`: the course's Trainium cluster is already set up, and you do not need it on your own machine.

### macOS: running the simulator in Docker

`neuronx-cc` has no macOS build, so on a Mac, run the simulator inside a Linux container. The `Dockerfile` in the `npu` directory builds an image with the simulator, PyTorch, and a copy of the starter code installed.

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and start it.
2. Build the image once, from the `npu` directory of your clone of the starter code:
   ```
   cd mp2/npu
   docker build --platform linux/amd64 -t mp2-nki-sim .
   ```
   This downloads the compiler and PyTorch, and produces an image of about 3.5 GB.
3. Run the tests on your own copy of the code:
   ```
   docker run --rm -it --platform linux/amd64 -v "$PWD":/mp2/npu mp2-nki-sim python3 test_harness.py --simulate
   ```

The `-v "$PWD":/mp2/npu` option mounts your `npu` directory into the container in place of the image's copy of the starter code. The container then runs the `conv_npu.py` you are editing on your Mac, and anything it writes lands in your directory. To get a shell in the container instead, leave off the command at the end:
```
docker run --rm -it --platform linux/amd64 -v "$PWD":/mp2/npu mp2-nki-sim
```
The image already sets `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, so there is no environment to activate.

The compiler is only built for x86-64 processors. On a Mac with an Apple Silicon (M-series) chip, Docker runs the image under x86-64 emulation, so the simulator runs slower than on an x86-64 machine.

### Running your kernel in the simulator

From the `npu` directory, run the test harness with the `--simulate` flag:
```
python3 test_harness.py --simulate
```
This runs the three correctness tests (small images, large images, and large images with bias) through the simulator. After them, the harness moves on to the performance tests, which need a real Trainium device, so in simulation mode it stops there with a Python traceback (a `KeyError`). That error is expected and does not mean your kernel is wrong; only the three correctness lines matter.

If your kernel allocates on-chip tensors with block dimensions, the compiler prints `DeprecationWarning: Block dimension is deprecated` lines. A block dimension is a tensor dimension placed before the partition dimension, such as the 8 in a tensor of shape `(8, par_dim(128), 512)`; this README explains them in [Direct Allocation](#direct-allocation) and uses them in its examples. Block dimensions still work in compiler version 2.22, so you can ignore these warnings.

**Which NKI API to use.** This assignment uses the NKI API in the `neuronxcc.nki` package, as the starter code does. AWS has since released a newer NKI API, imported with `import nki`, and the `latest` AWS documentation describes only that newer API. Do not mix the two. For documentation that matches this assignment, use the [Neuron SDK 2.26.1 NKI docs](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/nki/api/index.html).

### Running on the Trainium cluster

The performance tests and `neuron-profile` need a real Trainium accelerator, so the course runs a small cluster of Trainium nodes that the whole class shares through the [Slurm](https://slurm.schedmd.com/quickstart.html) job scheduler. You log in to a login node, which has no Trainium accelerator, and submit each test run as a job. The job waits in a queue until a Trainium node is free, then runs there. The cluster uses the same compiler version as the simulator, `neuronx-cc` 2.22. Get your kernel correct in the simulator first, and use the cluster for the performance tests and profiling.

#### Getting an account

You log in to the cluster with an SSH key instead of a password. If you do not have one yet, create it on your own machine, in a terminal on Linux or macOS or in PowerShell on Windows:
```
ssh-keygen -t ed25519
```
Press Enter to accept the defaults. This creates two files in the `.ssh` folder of your home directory: `id_ed25519`, your private key, and `id_ed25519.pub`, your public key. Never share your private key. If you already have a key, such as `id_rsa.pub`, you can use it instead.

Print your public key:
```
cat ~/.ssh/id_ed25519.pub
```
It is a single line that starts with `ssh-ed25519`. Submit it, with your NetID, through the [Trainium access form](https://docs.google.com/forms/d/e/1FAIpQLSeH-C6gpdMxFDf8RZvIkidAZBhPzflBO3SnJmt8fzc9y6jBjA/viewform). Your username on the cluster is your NetID. We will announce when accounts are ready.

#### Logging in

The login node accepts connections only from the campus network. When you are off campus, connect to the Illinois VPN first. Then log in with your NetID:
```
ssh <netid>@15.228.120.201
```
Your home directory on the cluster is private: other students cannot read your files. Keep your work there.

#### Getting your code onto the cluster

On the login node, clone the starter code:
```
git clone https://github.com/CS521-ml-compilers/mp2.git
cd mp2/npu
```
Then either edit `conv_npu.py` on the login node with `vim` or `nano`, or copy it there from your own machine after each change. To copy it, run this from the `npu` directory on your own machine:
```
scp conv_npu.py <netid>@15.228.120.201:mp2/npu/
```
If you keep your work in your own Git repository, make the repository private.

#### Running the tests on Trainium

From `mp2/npu` on the login node, submit the test harness as a job:
```
sbatch run_trainium.sh
```
`sbatch` prints the job's ID. Check on your jobs with `squeue --me`: a job's state is `PD` while it waits for a Trainium node and `R` while it runs. When your job no longer appears in the list, read its output, which has the correctness and performance results:
```
cat mp2-<job id>.out
```
To cancel a job, run `scancel <job id>`.

Keep these limits in mind:
- You can run one job at a time. If you submit another, it waits in the queue until the first one finishes.
- A job can run for at most 15 minutes. A complete test run takes about 3 minutes. A job that reaches the limit is stopped, and its output ends with a message that it was cancelled due to the time limit.
- Each job gets a whole Trainium node, so other students' jobs do not affect your timings. The whole class shares the nodes, so expect to wait in the queue at busy times, especially close to the deadline.
- The login node has no Trainium accelerator or Neuron software, so run your tests only through `sbatch`.

#### Profiling your kernel

To profile your kernel, pass `--profile` and a name to the test harness:
```
sbatch run_trainium.sh --profile conv
```
The job runs the same tests. The harness captures a profile of each performance test after measuring it, so profiling does not change your performance numbers. The end of `mp2-<job id>.out` then shows the MFU of the float32 and float16 tests, the numbers your write-up asks for.

The job also writes `conv_float32.pftrace` and `conv_float16.pftrace` to your `npu` directory: timelines of what each engine did while your kernel ran. To view one, copy it to your own machine and open it at [ui.perfetto.dev](https://ui.perfetto.dev). To copy it, run this on your own machine:
```
scp <netid>@15.228.120.201:mp2/npu/conv_float32.pftrace .
```
The job also leaves the raw captures, `conv_float32.neff` and `conv_float32.ntff` and their float16 counterparts, in your `npu` directory. Reading them takes `neuron-profile`, which is installed only on the Trainium nodes; the job has already used it to compute the MFU and write the timelines.

## Part 0: Getting familiar with Trainium, Neuron Core Architecture, and Neuron Kernel Interface

### Trainium Architecture Overview

First, let's get you acquainted with Trainium.

The `trn2.3xlarge` instance used in this assignment features a single Trainium2 device, which comprises eight NeuronCores (NeuronCore-v3) and 96 GiB of HBM (high-bandwidth memory). Each NeuronCore can be considered a standalone processing unit, which contains its own on-chip storage as well as a collection of specialized compute engines for performing 128x128 matrix operations (tensor engine), 128-wide vector operations (vector engine), etc. While each Trainium2 device has eight NeuronCores, in this assignment we will be writing kernels that execute on a single NeuronCore.

More details on the four distinct compute engines that exist in a NeuronCore can be found [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v3.html), and on the Trainium2 device [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html).

### Trainium Memory Hierarchy

On Trainium, the memory hierarchy consists of four levels: **host memory (DRAM)**, **device memory (HBM)**, and two fast on-chip memory types, **SBUF (State Buffer)** and **PSUM (Partial Sum Buffer)**. These levels are shown in the figure below.

<p align="center">
  <img src="../handout/memory_hierarchy.png" width=80% height=80%>
</p>

* __Host memory__, is the memory address space of the host machine, and is external to the Trainium device. You can think of host memory on Trainium as similar to host memory in a CUDA programming environment.
* __HBM__ is high-bandwidth memory located on the Trainium device. Moving data from host memory to HBM requires data transfer over the machine's PCIe interconnect. HBM serves as the device's primary memory, offering large storage (96 GiB on Trainium2). 
* __SBUF__ is on-chip storage on the NeuronCore. In comparison, SBUF is significantly smaller than HBM (28 MiB per NeuronCore on Trainium2) but offers much higher bandwidth (~20x than HBM). 
* __PSUM__ is a small, specialized memory (2 MiB) dedicated to holding matrix multiplication results produced by the tensor engine.

In Trainium, all computations require loading data from HBM into SBUF, which is accessible by all engine types. Intermediate data generated during kernel execution by the compute engines is also stored in SBUF. Once the computation is complete, the results are written back to HBM. 

<p align="center">
  <img src="../handout/neuron_core.png" width=40% height=40%>
</p>

Recall that in a system that features a traditional data cache, decisions about what data from off-chip memories is replicated and stored in on-chip storage are made by the cache (based on cache organization and eviction policies). Software loads data at a given memory address, and the hardware is responsible for fetching that data from memory and managing what data is stored in the cache for efficient future access. In other words, from the perspective of software correctness, the cache does not exist--it is a hardware implementation detail. 

In contrast, the memories available to a NeuronCore are *software managed*. This means that a software must explicitly move data to and from these memories using data movement commands. Either the programmer must explicitly describe data movement in their program, or the NKI compiler must analyze the application and generate the appropriate data movement operations.  Some of the biggest challenges of efficiently using the NeuronCore architecture involve efficiently orchestrating the movement of data through the machine.

### Neuron Kernel Interface (NKI) Programming Model:

The Neuron Kernel Interface (NKI) is a language and compiler for developing kernels that run on Trainium devices. NKI kernels are written in Python, and make use of three types of NKI operations:
1. **Loading data** from HBM to the on-chip SBUF.
2. **Computation** executed on the NeuronCore compute engines.
3. **Storing outputs** from SBUF back to HBM.

As an example, the following kernel defines how to perform vector addition using NKI. Note that the `@nki.jit` is a Python decorator that indicates that a function should be compiled to run on NeuronDevices, much like how the `__global__` function decorator in CUDA C++ designates that a function is to be compiled as a device-side function and run on the GPU.

Similar to how arguments to CUDA kernels are arrays in CUDA device global memory, the arguments to Python functions decorated with `@nki.jit`
 are tensors that reside in HBM accessible to the NeuronCore. In the following code, `a_vec` and `b_vec` are assumed to be length 128 vectors in HBM. (The code will not work for vectors that are larger than 128. We'll explain why shortly.)
```
@nki.jit
def vector_add_naive(a_vec, b_vec):
    
    # Allocate space for the output vector in HBM
    # it is the same dimensions as a_vec, with the same element type as a_vec
    out = nl.ndarray(shape=a_vec.shape, dtype=a_vec.dtype, buffer=nl.hbm)

    # Load the input vectors from HBM into variables stored in SBUF 
    a = nl.load(a_vec)
    b = nl.load(b_vec)

    # Add the input vectors
    res = nl.add(a, b)

    # Store the result into HBM
    nl.store(out, value=res)

    return out
```

In the code above...

- `nl.load` loads the two input vectors `a_vec` and `b_vec` from HBM to SBUF. Although it's not explicit in the code that `a` and `b` are variables backed by allocations in SBUF, you know this because they are variables returned by the `nl.load` function, which returns SBUF-based values.
- `nl.add` performs the vector addition of the two vectors.
- `nl.store` stores the results back to HBM.

<p align="center">
  <img src="../handout/sbuf_layout.png" width=60% height=60%>
</p>

**When looking at the code above, notice that NKI operations operate on tensors, not scalar values.** Specifically, the on-chip memories, SBUF and PSUM, store data that is arranged as 2D memory arrays. The first dimension of the 2D array is called the "partition dimension" `P`. The second dimension is referred to as the "free dimension" `F`.  NeuronCores are able load and process data along the partition dimension in parallel, *but the architecture also places a restriction that the size of the partition dimension is 128 or smaller.*  In other words, 
when loading a tensor from HBM to SBUF, the partition dimension of the tensor can be at most 128.  We will talk about the restrictions of the free dimension later.

As a result, in the code above, since `a_vec` and `b_vec` are 1D vectors, their only dimension is the partition dimension, and thus their size is limited to 128 elements.  In other words the code only works for vector sizes 128 or less.

#### Chunking Vectors to Parallelize Across 128 Compute Lanes

To fix the code to work for vectors with a size greater than 128, we need to load the vectors in chunks (subsets of the original tensor). 

```
@nki.jit
def vector_add_tiled(a_vec, b_vec):
    
    # Allocate space for the output vector in HBM
    out = nl.ndarray(shape=a_vec.shape, dtype=a_vec.dtype, buffer=nl.hbm)

    # Get the total number of vector rows
    M = a_vec.shape[0]
    
    # TODO: You should modify this variable for Step 1
    ROW_CHUNK = 1

    # Loop over the total number of chunks, we can use affine_range
    # because there are no loop-carried dependencies
    for m in nl.affine_range((M // ROW_CHUNK)):

        # Allocate row-chunk sized tiles for the input vectors
        a_tile = nl.ndarray((ROW_CHUNK, 1), dtype=a_vec.dtype, buffer=nl.sbuf)
        b_tile = nl.ndarray((ROW_CHUNK, 1), dtype=b_vec.dtype, buffer=nl.sbuf)
        
        # Load a chunk of rows
        a_tile[...] = nl.load(a_vec[m * ROW_CHUNK : (m + 1) * ROW_CHUNK])
        b_tile[...] = nl.load(b_vec[m * ROW_CHUNK : (m + 1) * ROW_CHUNK])

        # Add the row chunks together
        res = nl.add(a_tile, b_tile)

        # Store the result chunk into HBM
        nl.store(out[m * ROW_CHUNK : (m + 1) * ROW_CHUNK], value=res)
    
    return out
```

The above example breaks the vector rows into single-element chunks (the chunk size is 1 element of the vector---yes, this is inefficient, we'll come back to this in a second). This is achieved by indexing the vector using the standard Python slicing syntax `Tensor[Index:Index:...]`. More details regarding Tensor indexing in NKI can be found [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/programming_model.html#nki-tensor-indexing). 

In the code above [affine_range](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.language.affine_range.html) used here generates a sequence of numbers for loop iterators, similar to Python’s `range` function, but it requires that there are no loop-carried dependencies across iterations. Since there are no loop-carried dependencies across iterations, the NKI compiler can more aggressively optimize loop iterations to allow for increased pipelining across compute engines. In cases where loop iterations have dependencies, [sequential_range](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.language.sequential_range.html) should be used instead.

#### Improved Data Streaming

So far, we have been able to exploit the fact that the Vector Engine can perform computation with all 128 vector lanes in parallel, with each lane streaming a single element from/to a single SBUF/PSUM partition.

However, we can improve performance further by streaming more elements across the free dimension. The Neuron Compiler is responsible for converting each NKI load/store call to Direct Memory Access (DMA) transfer. You should think of a DMA transfer as a single asynchronous operation that moves a block of data from one storage location to another -- in this case between HBM and SBUF.  

The NeuronCore has 16 DMA engines that can all work on different data transfer operations in parallel. The caveat is that there is an overhead cost when setting up a DMA transfer and assigning DMA engines to work on them. In order to reduce this setup overhead, efficient implementations should aim to move a large amount of data in each DMA transfer to amortize DMA transfer overhead.

Although the first dimension (partition dimension) of a SBUF tensor can be no greater than 128, the second dimension for a single SBUF vector instruction can be up to 64K elements. This means that it is possible to use a single instruction to load 128 * 64k = 8192k elements from HBM to SBUF. Furthermore, we can perform vector addition on two 8192k element SBUF tiles in a single arithmetic instruction. Therefore, rather than performing a `nl.load` operations (and equivalently a hardware DMA transfer request) for each 128 element chunk of a vector, we should should instead move many 128-row chunks with each DMA transfer request. This streamlined approach allows us to amortize the setup time required for transferring data.

In order to improve DMA transfer overhead, we will need to reshape our vectors so they are two-dimensional tiles, rather than linearized arrays. The goal is to turn a single-dimension vector into a dense 2D matrix. NumPy comes with a built-in [reshape function](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) allowing you to reshape arrays into the shape of your choosing. 

<p align="center">
  <img src="../handout/non_reshaped_DMA.png" width=48% height=48%>
  <img src="../handout/reshaped_DMA.png" width=48% height=48%>
</p>


Take a look at `vector_add_stream`, which extends `vector_add_tiled` so that there are less DMA transfers:
```
@nki.jit
def vector_add_stream(a_vec, b_vec):

    # Get the total number of vector rows
    M = a_vec.shape[0]

    # TODO: You should modify this variable for Step 1
    FREE_DIM = 2

    # The maximum size of our Partition Dimension
    PARTITION_DIM = 128

    # The total size of each tile
    TILE_M = PARTITION_DIM * FREE_DIM

    # Reshape the the input vectors
    a_vec_re = a_vec.reshape((M // TILE_M, PARTITION_DIM, FREE_DIM))
    b_vec_re = b_vec.reshape((M // TILE_M, PARTITION_DIM, FREE_DIM))

    # Allocate space for the reshaped output vector in HBM
    out = nl.ndarray(shape=a_vec_re.shape, dtype=a_vec_re.dtype, buffer=nl.hbm)

    # Loop over the total number of tiles
    for m in nl.affine_range((M // TILE_M)):

        # Allocate space for a reshaped tile
        a_tile = nl.ndarray((PARTITION_DIM, FREE_DIM), dtype=a_vec.dtype, buffer=nl.sbuf)
        b_tile = nl.ndarray((PARTITION_DIM, FREE_DIM), dtype=a_vec.dtype, buffer=nl.sbuf)

        # Load the input tiles
        a_tile[...] = nl.load(a_vec_re[m])
        b_tile[...] = nl.load(b_vec_re[m])

        # Add the tiles together
        res = nl.add(a_tile, b_tile)

        # Store the result tile into HBM
        nl.store(out[m], value=res)

    # Reshape the output vector into its original shape
    out = out.reshape((M,))

    return out
```

#### Direct Allocation

As discussed above, a multi-dimensional NKI tensor requires one of its dimensions to be mapped to the partition dimension (`P`) of physical memory. We also define an NKI Tile as any NKI Tensor with the first dimension being the partition dimension. 

To introduce NKI allocation API, let’s define the block (`B`) dimension as any dimension before the partition dimension in a NKI Tensor. Therefore, a NKI Tensor has three types of dimensions: `(B, P, F)`. Note, a NKI Tensor can have one or more dimensions in both `B` and `F`, but there can only be one dimension in `P` due to Neuron ISA requirements. The block dimension effectively describes how many `(P, F)` NKI Tiles the tensor has, which commonly corresponds to how many NKI API compute API invocations we need to process the entire tensor.

In fact, the block dimensions are considered **logical** in NKI - the number of `(P, F)` NKI Tiles suggested by dimension (`B`) may not match the number of physical tiles allocated. The Neuron Compiler analyzes the code and uses a heuristic-driven allocator to determine the number of physical tiles allocated to each tensor. Given the amount of parallelism available in hardware and the complex parallel programs seen in common machine learning workloads, the heuristic-based memory allocator in Neuron Compiler may not yield the optimal allocation decisions. Bad allocation decisions typically lead to sub-optimal engine parallelism and/or on-chip memory over-subscription causing excessive spills of intermediate data to device memory. 

For example, consider a scenario where the block dimension size of `nki_tensor` is 2, yet only a single physical tile, `T`, is allocated in `sbuf`.
```
   i_block = 0
1. nl.load(nki_tensor[0, :, :]) => write T
2. nl.exp(nki_tensor[0, :, :])  => read T

   i_block = 1
3. nl.load(nki_tensor[1, :, :]) => write T
4. nl.exp(nki_tensor[1, :, :])  => read T
```

In this example, there is only one logical tile in `nki_tensor` that is alive at one time. However, in NeuronCore, `nl.load` and `nl.exp` are executed using two independent resources: DMA Engine and Scalar Engine. In this serialized execution, there is no instruction parallelism achieved between these engines. There is potential for performance optimization by allocating more physical tiles.

With [NKI direct allocation API](https://awsdocs-neuron-staging.readthedocs-hosted.com/_/sharing/PsLqSocIDcWDsjcHd3ql7JCqOhou3AKO?next=/en/nki_docs_2.21_beta_class/general/nki/nki_direct_allocation_guide.html#nki-direct-allocation-guide), programmers can bypass the compiler allocator and take full control of memory allocation in SBUF/PSUM for NKI Tensors. NKI provides two kinds of direct allocation API, `nisa.sbuf.alloc()` and `nisa.sbuf_mod_alloc()`. The first one requires programmers to define an allocation algorithm from scratch, while the second one invokes a pre-defined modulo allocation scheme in Neuron Compiler. In this tutorial, we will focus on the `nisa.sbuf.mod_alloc()` API, which is already powerful enough.
Modulo allocation works as follows. Suppose that we allocate two physical tiles for a tensor with a logical shape of `(8, par_dim(128), 512)`. The eight logical tiles are assigned to the two physical tiles by taking a modulo of two on the logical tile index. Therefore, logical tiles with index (0, ), (2, ), (4, ), (6, )share the same physical tile, while logical tiles (1, ), (3, ), (5, ), (7, ) share the other physical tile.  

The `nisa.sbuf.mod_alloc` API takes four input parameters:
1. `base_addr` indicates the starting byte offset within each SBUF partition of the physical tiles.
2. `base_partition` indicates the starting SBUF partition of the physical tiles.
3. `num_par_tiles` indicates the number of physical tiles to be allocated along the partition dimension of SBUF. This is only applicable for tiles that use fewer than 64 partitions per ISA constraints.
4. `num_free_tiles` indicates the number of physical tiles to be allocated along the free dimension of SBUF.

Take a look at `vector_add_direct_allocation`, which uses `nisa.sbuf.mod_alloc` to allocate physical tiles for input and output tensors.

```
@nki.jit
def vector_add_direct_allocation(a_vec, b_vec):
    # Create output tensor in HBM 
    out = nl.ndarray(a_vec.shape, dtype=a_vec.dtype, buffer=nl.shared_hbm)
    
    # Define constants for free dimension, physical tile count, and partition dimension
    FREE_DIM = 1000
    FREE_DIM_TILES = 4
    PARTITION_DIM = 128

    # Get the total number of vector rows
    M = a_vec.shape[0]

    # Define the size of each tile
    TILE_M = PARTITION_DIM * FREE_DIM

    # Reshape the the input vectors
    a_vec = a_vec.reshape((M // TILE_M, PARTITION_DIM, FREE_DIM))
    b_vec = b_vec.reshape((M // TILE_M, PARTITION_DIM, FREE_DIM))
    out = out.reshape((M // TILE_M, PARTITION_DIM, FREE_DIM))

    # Get the total number of tiles
    N_TILES = M // TILE_M

    # Initialize the starting byte offset for a_tensor
    current_offset = 0
    
    # Allocate space for the entire reshaped a_tensor with 4 physical tiles
    a_tile = nl.ndarray((N_TILES, nl.par_dim(PARTITION_DIM), FREE_DIM), dtype=a_vec.dtype,
            buffer=ncc.sbuf.mod_alloc(base_addr=current_offset, num_free_tiles=(FREE_DIM_TILES,)))

    # Increment the starting byte offset for b_tensor based on tile and feature size
    current_offset += FREE_DIM_TILES * FREE_DIM * 4
    
    # Allocate space for the entire reshaped b_tensor with 4 physical tiles
    b_tile = nl.ndarray((N_TILES, nl.par_dim(PARTITION_DIM), FREE_DIM), dtype=b_vec.dtype,
            buffer=ncc.sbuf.mod_alloc(base_addr=current_offset, num_free_tiles=(FREE_DIM_TILES,)))

    # Increment the starting byte offset, and allocate space for the entire reshaped out_tensor
    current_offset += FREE_DIM_TILES * FREE_DIM * 4
    res_tile = nl.ndarray((N_TILES, nl.par_dim(PARTITION_DIM), FREE_DIM), dtype=out.dtype,
            buffer=ncc.sbuf.mod_alloc(base_addr=current_offset, num_free_tiles=(FREE_DIM_TILES,)))

    # Loop over the total number of tiles
    for m in nl.affine_range(N_TILES):
        
        # Load the input tiles
        a_tile[m] = nl.load(a_vec[m])
        b_tile[m] = nl.load(b_vec[m])

        # Add the tiles together
        res_tile[m] = nisa.tensor_tensor(a_tile[m], b_tile[m], op=nl.add)
        
        # Store the result tile into HBM
        nl.store(out[m], value = res_tile[m])

    # Reshape the output vector into its original shape
    out = out.reshape((M, 1))

    return out
```

#### Learning to Use Neuron-Profile

To get a detailed analysis of the performance of an application running on a NeuronCore, you will need to use the profiling tool for NeuronDevices: `neuron-profile`. Feel free to learn more about `neuron-profile` functionality from the [user guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html) and interesting performance metrics for NKI kernels from the [NKI performance guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/nki_perf_guide.html).

`neuron-profile` only runs on Trainium, not in the simulator. [Profiling your kernel](#profiling-your-kernel) explains how to profile your kernel on the course's cluster and view the results.

## Part 1: Implementing a Convolution Layer

Now that you’ve learned how to efficiently move data on a NeuronCore, it is time to program an actual Trainium kernel yourself. In this section, your task is to implement a kernel that performs convolution. As we discussed in class, the convolution operation is a fundamental component of modern Convolutional Neural Networks (CNNs), which are extensively used for computer vision tasks.

### Matrix Operations on a NeuronCore

Before you begin, we will demonstrate how to perform matrix operations on a NeuronCore. As discussed earlier, a NeuronCore is equipped with various compute engines, each optimized for specific types of arithmetic operations. In Part 0, you worked with the Vector Engine, which specializes in vector operations like vector addition. However in Part 1, you not only need to perform vector operations, but you will also need to perform matrix operations. The Tensor Engine on Trainium is specifically designed to accelerate these matrix operations, such as matrix multiplication and matrix transpose. 

<p align="center">
  <img src="../handout/tensor_engine.png" width=60% height=60%>
</p>

The above image depicts the architecture of the Tensor Engine. The Tensor Engine is built around a 128x128 [systolic processing array](https://gfxcourses.stanford.edu/cs149/fall24/lecture/hwprog/slide_10) which streams matrix data input from SBUF (on-chip storage) and writes the output to PSUM (also on-chip storage). Like SBUF, PSUM is fast on-chip memory, however it is much smaller than SBUF (2 MiB vs 28 MiB) and serves a dedicated purpose of storing matrix multiplication results computed by the Tensor Engine. The Tensor Engine is able to read-add-write to every address in PSUM. Therefore, PSUM is useful when executing large matrix multiplications in a tiled manner, where the results of each matrix multiply are accumulated into the same output tile. 

Recall that the Vector Engine has the capability to operate on SBUF tiles of size (128, 64k). However, the Tensor Engine contains unique SBUF tile size constraints which differ to that of the Vector Engine. Suppose we want the Tensor Engine to perform the matrix multiplication C = A x B, where A and B are located in SBUF, and the result C is stored in PSUM. Trainium imposes the following constraints. 
  - Matrix A - the left-hand side tile - can be no bigger than (128, 128)
  - Matrix B - the right-hand side tile - can be no bigger than (128, 512).
  - The output tile C in PSUM is restricted to a size of (128, 512).

### An NKI Matrix Multiplication Kernel

Given the constraints of the Tensor Engine, implementing matrix multiplication for arbitrary matrix dimensions on Trainium requires tiling the computation so it is performed as a sequence of matrix multiplications on fixed-size tiles. (This is similar to how vector addition in part 0 was tiled to work for large input vector sizes). The example below, which we copied from the [NKI tutorials](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/tutorials.html), demonstrates how to implement matrix multiplication using a tiled approach, where the tiles are sized to meet the Trainium Tensor engines tile size constraints. Note: a description of the code is provided after the code listing.

```
def nki_matmul_tiled_(lhsT, rhs, result):
  """NKI kernel to compute a matrix multiplication operation in a tiled manner"""

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # Maximum free dimension of the stationary operand of general matrix multiplication on tensor engine
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128

  # Maximum partition dimension of a tile
  TILE_K = nl.tile_size.pmax  # 128

  # Maximum free dimension of the moving operand of general matrix multiplication on tensor engine
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      # Allocate a tensor in PSUM
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        # Declare the tiles on SBUF
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

        # Load tiles from lhsT and rhs
        lhsT_tile[...] = nl.load(lhsT[k * TILE_K:(k + 1) * TILE_K,
                                      m * TILE_M:(m + 1) * TILE_M])
        rhs_tile[...] = nl.load(rhs[k * TILE_K:(k + 1) * TILE_K,
                                    n * TILE_N:(n + 1) * TILE_N])

        # Accumulate partial-sums into PSUM
        res_psum += nl.matmul(lhsT_tile[...], rhs_tile[...], transpose_x=True)

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nl.store(result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N],
               value=res_sb)
```

Let's break down the components of the kernel which computes the matrix multiply: `result = lst x rhs`.

  - Input Tensors:
      - `lhsT` is the left-hand-side matrix. But the matrix is provided in a __transposed format__ with shape `[K,M]`, where both `K` and `M` are multiples of 128.
      - `rhs` is the right-hand-side matrix, with shape `[K,N]`, where `K` is a multiple of 128, and `N` is a multiple of 512.
      - `result` is the output matrix of shape `[M,N]`
      - In matrix multiplication, the **contraction dimension** refers to the column dimension of the left-hand matrix and the row dimension of the right-hand matrix. For example, say 
        we have the following matrix multiplication: `A x B = C`. The matrix `A` has shape  
        `[M, N]` and the matrix `B` has shape `[N, M]`. The shape of `C` is then `[M, M]`. 
        Thus, the dimensions that were eliminated was the column dimension of `A` and the row dimension of `B`. Please note that in the `nki_matmul_tiled_` example above, the 
        matrix is in transposed form and is transposed again in the `nl.matmul` call. Thus, the matrix multiplication which is being performed is  
        `A^T x B = C`. In the transposed case, the 
        contraction dimension is the row dimension of `A` and the row dimension of `B` due to the fact that `A` is transposed before multiplication.
  - Tile Dimensions:
      - The tile sizes are set based on the constraints of the tensor engine matrix multiplication operation, as described [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.isa.nc_matmul.html).
        - `TILE_M`: 128 — Tile size for the `M` dimension.
        - `TILE_K`: 128 — Tile size for the `K` dimension.
        - `TILE_N`: 512 — Tile size for the `N` dimension.
  - Looping Over Tiles:
      - The kernel uses `affine_range` loops to iterate over tiles along the `M` and `N` dimensions of the `result` matrix.
      - For each output tile of shape `(TILE_M, TILE_N)`, it allocates a temporary partial sum tensor `res_psum` in PSUM memory.
  - Loading Tiles:
      - For each output tile, tiles of `lhsT` and `rhs` are loaded into the on-chip SBUF memory for efficient access.
      - `lhsT_tile` is loaded with a slice of shape `[TILE_K, TILE_M]`, and `rhs_tile` is loaded with a slice of shape `[TILE_K, TILE_N]`.
  - Matrix Multiplication:
      - A partial matrix multiplication is performed using the loaded tiles and partial results are accumulated into `res_psum`.
  - Storing Results:
      - Once the tiles for a given result block are fully computed, the partial sums in `res_psum` are copied to SBUF and cast to the required data type.
      - The final result is stored back into the `result` tensor at the corresponding position.

In summary, this tiled implementation handles large matrix dimensions by breaking them into hardware-compatible tile sizes. It leverages specialized memory buffers (i.e., PSUM) to minimize memory latency and optimize matrix multiplication performance. You can read more about NKI matrix multiplication [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/tutorials/matrix_multiplication.html).

### Convolution Layer Overview

Let’s now turn our focus to the convolution layer. Recall the [convolution operation](https://gfxcourses.stanford.edu/cs149/fall24content/media/dnneval/10_dnneval.pdf). It involves sliding a filter across an __input feature map__, where at each position the filter interacts with the overlapping input region. In each overlapping region, element-wise multiplications are performed between the filter weights and the input region region values. The results of these element-wise multiplications are then added together, producing a single value for the corresponding position in the output feature map. This process captures local spatial patterns and relationships among neighboring features.

<p align="center">
  <img src="../handout/convolution.png" width=55% height=55%>
</p>

The input feature map often consists of multiple channels. For example, an image usually contains three RGB channels (red, green, and blue). In this case, instead of only computing a weighted sum over the 2D spatial region, the convolution computes the weighted sum of both the 2D spatial region and channel depth. The image below depicts an example of a convolution layer being performed on a 32x32 input image with three RGB channels. In the image, a 5x5x3 filter is applied on the 32x32x3 image to produce a 28x28x1 output feature map.

<p align="center">
  <img src="../handout/cs231n_convolution.png" width=55% height=55%>
  <br>
  <em>Source: CS231N https://cs231n.stanford.edu/slides/2024/lecture_5.pdf </em>
</p>

__As seen in the image, each filter produces a single channel of output.__ To generate multiple output channels, multiple filters are applied to the input featuer map. In addition to this, each convolution filter also contains a scalar bias value that is to be added to each weighted sum. 

The input and output of the convolution operator can be summarized as follows (ignoring bias for now):

<p align="left">
  <img src="../handout/conv2d_summary.png" width=50% height=50%>
</p>

Additionally, a [convolution layer](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html) can take in additional hyper-parameters such as padding and stride in addition to an input feature map, filter weights, and scalar bias. However, we have *simplified the constraints of your convolution* to make implementation easier for you. You need **only to support a stride of 1**, and you do **not have to worry about padding** as we will pad the input feature map for you before it is passed into your kernel.

### Mapping Convolution to Matrix Multiplication

Now, our objective is to map the convolution operator onto thee high-performance matrix operations supported by the Trainium's Tensor engine.  To do this, we can compare the mathematical formulation of convolution with matrix multiplication.

**Conv2D:**

<p align="center">
  <img src="../handout/conv2d_formula.png" width=65% height=65%>
</p>

**Matrix Multiplication:**

<p align="center">
  <img src="../handout/matmul_formula.png" width=25% height=25%>
</p>

In class we discussed one way to convert convolution with many filters into a single large matrix multiplication.  We'll do the same thing here, but take a different approach that yields an efficient implementation on Trainium.  In this approach the convolution operation is formulated as a series of independent matrix multiplications. A visual illustration of this formulation is shown below.

<p align="center">
  <img src="../handout/conv2d_matmul_diagram.png" width=100% height=100%>
</p>

In this approach, the height and width dimensions of the input feature map are flattened into a single dimension, reshaping the input to `(Height × Width) × Input Channels​`. This reshaped input is then multiplied by each position of the filters, where `i` and `j` respectively range from `0` to `Filter Height - 1` and from `0` to `Filter Width - 1`. Each filter slice has a shape of `Input Channels × Output Channels`, and the resulting matrix multiplication contracts along the `Input Channels` dimension. To align the input with each filter slice, the input must be shifted by an offset corresponding to the filter’s current position `(i, j)`. The results of these matrix multiplications are accumulated to produce the output tensor.

Below is the pseudocode for the described algorithm:
```
- Have the input image with shape (Input Channels, Image Height * Image Width)
- Have the filter weights with shape (Filter Height, Filter Weight, Input Channels, Output Channels)
- Initialize the output to appropriate shape of (Output Channels, Output Height * Output Width)

# Iterate over the filter height
for i in range(Filter_Height):
    # Iterate over the filter width
    for j in range(Filter_Width):

        # Shift the Input tensor by (i, j) to align with the filter's current position
        input_shifted = shift(input, (i, j))

        # Perform matrix multiplication between the input and the filter slice
        output += matmul(transpose(weight[i,j,:,:]), input_shifted)
```

> Do note that this just an algorithmic description, and the purpose of this assignment is for you to figure out to map this algorithmic description to an efficient implementation on this hardware!

### Implementing an NKI kernel for Convolution
You will implement an NKI kernel for the convolution layer. Below, we will outline the detailed specifications and requirements for your convolution layer.

Your convolution kernel takes in the following parameters:
  - `X` - A batch of input images. `X` has shape `(Batch Size, Input Channels, Input Height, Input Width)`. You are guaranteed that `Input Channels` will be a multiple of 128.
  - `W` - The convolution filter weights. `W` has shape `(Output Channels, Input Channels, Filter Height, Filter Width)`. You are guaranteed that `Filter Height == Filter Width`. You are also guaranteed that `Output Channels` is a multiple of 128. Moreover, you can assume that the size of the weights would always be such that it can completely fit inside SBUF.
  - `bias` - The convolution filter biases. `bias` has shape `(Output Channels)`

Feel free to use the [reference](https://gfxcourses.stanford.edu/cs149/fall24/lecture/dnneval/slide_43) on a convolution layer implementation as a starting point, where `INPUT_DEPTH` is synonymous with `Input Channels` and `LAYER_NUM_FILTERS` is synonymous with `Output Channels` in our naming scheme. Note that the input parameters to your convolution kernel have different shapes than depicted in the convolution course slides. You are free to reshape the inputs into whatever shapes you desire by using the [NumPy reshape method](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) just as was done in `vector_add_stream kernel` from Part 0. We have also given you the NumPy implementations of a convolution layer in `conv_numpy.py`. The NumPy implementations should give you a general outline of the programming logic for each layer. It might be a good exercise to think about how you would be able to fuse the NumPy implementations into a single layer, which is what you will do in your kernel. Feel free to look over the [NKI tutorials](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/tutorials.html) to learn more about additional optimizations or other API functions. You can also view the [API Reference Manual](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/index.html) to see all of the API functions that are available and their usage.

### What You Need To Do
For this part of the assignment, focus exclusively on the file `conv_npu.py`. We've provided basic starter code; your task is to complete the implementation of the Conv2D kernel within the function `conv2d`.

Prioritize Correctness First. Before optimizing for performance, ensure your implementation is correct. While you may choose to implement things differently, we recommend starting by creating a kernel that works for small image sizes. Then, once your kernel works for small images, extend its functionality to handle images that are too large to fit entirely in the SBUF buffer. Following that, incorporate bias addition. Proceed to optimize performance once you achieve correctness.

Use the test harness script provided to validate your implementation. To run the tests on Trainium, submit them as a job from the `npu` directory on the course's cluster, as [Running the tests on Trainium](#running-the-tests-on-trainium) describes:
```
sbatch run_trainium.sh
```

To debug your implementations, run the test harness with the `--simulate` flag. This wraps your implementation with a call to `nki.simulate_kernel()`: you can read more about it [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.simulate_kernel.html#nki.simulate_kernel). When running in simulation mode, you can insert calls to `nl.device_print()` to print intermediate values of the device tensors. This can help identify potential bugs.

The test harness will run correctness tests first, and run performance checks next. A full-credit solution must achieve performance within 150% of the reference kernel while maintaining correctness. It will invoke your kernel with input tensors having data types float32 and float16: with the performance requirements for float16 being more strict. Make sure you write your kernels keeping this in mind!

Students also need to submit a write up briefly describing their implementations. Also describe how you went about optimizing your implementation. Make sure to profile your implementation, and report the achieved MFU, with both `bfloat16` and `float32` data types. [Profiling your kernel](#profiling-your-kernel) explains how to profile your kernel with the test harness's `--profile <profile_name>` flag and where to find its MFU.

## Grading Guidelines

For the correctness test, we use two types of images. The first type is a small image with dimensions of 32×16. The second type is a large image with dimensions of 224×224, which exceeds the capacity of the SBUF and cannot fit within it.

For the performance test, we evaluate the performance using float16 and float32 precision. We will compare the performance of your program with the reference solution. You will pass the test if your p99 latency is within 150% of the reference latency. Please note that the performance requirement in the `test_harness.py` is already 150% of the reference latency.

## FAQs

Please refer to an advanced FAQ here: https://github.com/EmilyWebber/nki-samples/blob/main/contributed/advanced_FAQs.md. You can also open a support ticket at https://github.com/aws-neuron/aws-neuron-sdk/issues/new if you have any other questions, thanks!
