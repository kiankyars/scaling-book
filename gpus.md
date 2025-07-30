---
layout: distill
title: "How to Think About GPUs"
description: "After talking so much about TPUs, it's probably worth taking a look at what so much of the rest of the world uses: NVIDIA GPUs. This will be a deep-dive into both the chip and networking levels of a modern NVIDIA ML GPU (e.g. H100 or B100) and what kinds of LLM parallelism they allow. You are encouraged to read this after the rest of the book."
date: 2025-07-25
future: true
htmlwidgets: true
hidden: false

section_number: 12

previous_section_url: 
previous_section_name: ...

next_section_url:
next_section_name: ...

bibliography: main.bib

giscus_comments: true

authors:
  - name: To Be Determined
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: Google DeepMind

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: What Is a GPU?
  - subsections:
    - name: "Summary of GPU Specs"
    - name: "Grace Hopper"
    - name: GPUs vs. TPUs at the Chip Level
    - name: Worked Problems
  - name: Networking
  - subsections:
    - name: Node Level
    - name: Worked Problems
    - name: Beyond the Node Level
  - name: How Do Collectives Work on GPUs?
  - subsections:
    - name: Within a node
    - name: In network reductions
    - name: Beyond the node level
    - name: Worked problems
  - name: "Takeaway for LLM Scaling on GPUs"
  - subsections:
    - name: Worked problems
  - name: "How does this change with B100 and the GB200 NVL72?"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

GPUs started as specialized hardware for rendering video games, but since the explosion of AI demand in the 2010s, they’ve started looking more and more like dedicated matrix multiplication machines – in other words, just like TPUs. Both TPUs and GPUs act like matrix multiplication accelerators attached to a CPU. They differ in two key respects: how they’re networked together, and how much responsibility they place on the software to do the right thing (*hint: TPUs need the compiler to do a lot more work*).

## What Is a GPU?

A modern GPU (e.g. H100, B100) is basically a bunch of compute cores that specialize in matrix multiplication (called Streaming Multiprocessors or SMs) all connected to a stick of fast memory (called DRAM or HBM). Here’s a diagram:

{% include figure.liquid path="assets/gpu/gpu-diagram.png" class="img-fluid" caption="<b>Figure:</b> the basic components of a modern NVIDIA GPU. The diagram shows the SMs containing a set of Tensor Cores and Warp Schedulers (containing CUDA cores), SMEM, the shared L2 cache, and the main GPU memory (HBM)." %}

Unlike a TPU, which has at most 2 Tensor Cores, **a modern GPU has more than 100 of these SMs** (132 on an H100). Consequently, each of these SMs is much less powerful than a TPU TensorCore but the system overall is more flexible. Each SM is more or less totally independent and so a GPU can do hundreds of tasks at once, although all SMs share a 50MB L2 cache and a large amount of DRAM (80GB on H100, 192 on B100), as shown above. Here’s a more detailed view of an B100 SM:

{% include figure.liquid path="assets/gpu/broadwell-sm.png" class="img-small" caption="<b>Figure:</b> a deeper look at a single Broadwell (B100) SM, showing the four SM subpartitions and their CUDA cores, along with 4 TensorCores and some auxilliary units." %}

Each SM is broken up into 4 identical quadrants, which NVIDIA calls "SM subpartitions", each containing a Tensor Core, 16k 32-bit registers, and a set of SIMD lanes NVIDIA calls "CUDA Cores". The core component of each partition is arguably the Tensor Core, which performs matrix multiplications and makes up the vast majority of FLOPs/s, but like a TPU there are a few other components worth noting.

* **CUDA Cores:** each subpartition contains a set of ALUs called **CUDA Cores** that do SIMD vector arithmetic, much like a TPU’s VPU. Each subpartition contains 32 fp32 cores (and a smaller number of int32 and fp64 cores) which function as a single SIMD unit performing the same vector operation in each cycle. **Despite NVIDIA’s dubious naming, you should think of these as lanes in a 32-wide SIMD unit performing the same vector operation in each cycle.**

    * Each CUDA core (SIMD lane) of a particular precision within a subpartition executes in lockstep, so per-cycle, each lane must perform the same work, just like the TPU’s VPU.
    * For ML models, CUDA cores are typically used to perform pointwise operations like ReLUs, vector additions, norms, and other non-matmul work.
    * The CUDA cores within a subpartition are controlled by a dispatch unit called a **warp scheduler**, which acts like the controller of the SIMD unit. Each warp scheduler runs a bit like a multi-threaded CPU, in the sense that it can run many programs (called **warps**) concurrently (up to 16 per subpartition) but only ever executes instructions from a single program in each clock cycle. The hardware automatically switches between active warps to hide I/O operations like memory loads.
    * Each warp scheduler has its own register file (16,384 32-bit words on H100/B100, for a total of `4 * 16384 * 4 = 256kB` of register memory per SM). Each CUDA core can only access up to 256 registers, so although we can schedule up to 16 “resident warps” per warp scheduler, if each core uses 256 registers, you can only fit 2 at a time.

* **Tensor Core (TC):** each subpartition has its own Tensor Core, which is a dedicated matrix multiplication unit like a TPU MXU. The Tensor Core represents the vast majority of the GPUs FLOPs/s (e.g. on an H100, we have 990 bf16 FLOP/s compared to just 66 FLOPs/s from the CUDA cores).

    * An H100 has a peak bfloat16 matmul throughput of 990 bfloat16 TFLOPs per second, so each SM can do about 7.5TFLOPs peak. Since each SM has 4 TCs and each SM runs at a peak frequency of 1.76GHz, each TC can do roughly `7.5e12 / 1.76e9 / 4 ~ 1024` bf16 FLOPs/s, so roughly an `8x8x8` matmul each cycle.
    * Each GPU generation has gotten larger Tensor Cores as matrix multiplication compute has become more and more important ([good article on this](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)).
    * Like TPUs, GPUs can do lower precision matmuls at higher throughput. An H100 can do 990 bf16 TFLOPs/s and 1979 fp8 TFLOPs/s, around 2x. This means if you can effectively train or serve a model at lower precision, you will see significant gains in performance.
    * Historically, GPU Tensor Cores loaded their inputs from SMEM or register memory, but as the TCs have gotten bigger, it’s become harder to fit the full inputs. B100/B200s introduce a new kind of on-chip memory called Tensor Memory (or TMEM) which is used to store inputs to matmuls done in the TCs.

Beyond the compute units, GPUs have a hierarchy of memories, the largest being HBM (the main GPU memory), and then a series of smaller caches (L2, L1, TMEM, register memory).

* **SMEM (L1 Cache):** each SM has its own small on-chip cache, called SMEM, which can either be programmer controlled as “shared memory” or used by the hardware as an on-chip cache.

    * SMEM is heavily used for storing inputs to TC matmuls and for storing activations while they’re being processed, so we don’t need to load fully from DRAM.
    * Because SMEM is so much smaller than TPU VMEM, it’s harder to fit whole layers of a model into on-chip memory.

* **L2 Cache:** all SMs share a relatively large \~50MB L2 cache used to reduce main memory accesses.

    * This is similar in size to a TPU’s VMEM, with higher bandwidth to the TCs than HBM, but isn’t programmer controlled and is **much** slower. This leads to a bit of “spooky action at a distance” where the programmer needs to modify memory access patterns to ensure the L2 cache is well used.
    * NVIDIA does not publish the L2 bandwidth for their chips, but it’s been measured to be about 5.5TB/s, or roughly 1.6x the HBM bandwidth. By comparison, a TPU’s VMEM is 2x larger *and* has much more bandwidth (around 40TB/s).

* **HBM:** the main GPU memory, used for storing model weights, gradients, activations, etc.

    * The HBM size has increased a lot from 32GB in Volta to 192GB in Blackwell (B100).
    * The bandwidth from HBM to the CUDA Tensor Core is called HBM bandwidth or memory bandwidth, and is about 3.35TB/s on H100.

Here’s a helpful cheat sheet comparing GPU and TPU components:

|              GPU              |     TPU     |              What is it?              |
| :---------------------------: | :---------: | :-----------------------------------: |
| Streaming Multiprocessor (SM) | Tensor Core | Core “cell” that contains other units |
|        Warp Scheduler         |     VPU     |      SIMD vector arithmetic unit      |
|           CUDA core           |  VPU lane   |            SIMD ALU “lane”            |
|        SMEM (L1 Cache)        |    VMEM     |       Fast on-chip cache memory       |
|          Tensor Core          |     MXU     |      Matrix multiplication unit       |
|             DRAM              |     HBM     |  High bandwidth high capacity memory  |

### Summary of GPU Specs

Here is a summary of GPU specs for recent models:

|  GPU  | Generation |  SMs  | SMEM per SM (kB) | L2 Cache (MB) | Clock Speed (GHz) | DRAM (GB) | DRAM BW (TB/s) | BF16 TFLOPs | FP8 TFLOPs | FP4 TFLOPs |
| :---: | :--------: | :---: | :--------------: | :-----------: | :---------------: | :-------: | :------------: | :---------: | :--------: | :--------: |
| V100  |   Volta    |  80   |        96        |       6       |     1.25/1.38     |    32     |      0.9       |      —      |     —      |     —      |
| A100  |   Ampere   |  108  |       192        |      40       |     1.10/1.41     |    80     |      2.0       |     312     |     —      |     —      |
| H100  |   Hopper   |  132  |       256        |      50       |     1.59/1.76     |    80     |      3.35      |     990     |    1979    |     —      |
| H200  |   Hopper   |  132  |       256        |      50       |     1.59/1.76     |    141    |      4.8       |   (same)    |   (same)   |     —      |
| B100  | Blackwell  |  144  |       256        |      50       |     1.67/1.83     |    192    |       8        |    1800     |    3500    |    7000    |
| B200  | Blackwell  |   ?   |       256        |      50       |         ?         |    192    |       8        |    2250     |    4500    |    9000    |

All generations have 256kB of register memory per SM. Blackwell also adds 256kB of TMEM per SM. Some specs depend slightly on the precise version of the GPU.

### Grace Hopper

NVIDIA also sells GH200 and GB200 systems which pair some number of GPUs with a Grace Hopper CPU. For instance, a GH200 has 1 H200 and 1 GH CPU, while a GB200 system has 2 B200s and 1 GH CPU. An advantage of this system is that the CPU is connected to the GPUs using a full bandwidth NVLink connection (called NVLink C2C), so you have very high CPU to GPU bandwidth, useful for offloading parameters. In other words, for any given GPU, the bandwidth to reach host memory is identical to reaching another GPU’s HBM.

### GPUs vs. TPUs at the Chip Level

As you’ve hopefully noticed, GPUs and TPUs look quite similar at a chip level. They both have matmul accelerators, SIMD vector units, and cache memory. One key difference is that TPUs have 1-2 big Tensor Cores, while GPUs have hundreds of small SMs. Likewise, each Tensor Core has 1 big VPU with 4096 ALUs while GPUs have an H100 has 132 * 4 = 528 small independent SIMD units.

Here is a 1:1 comparison of GPUs to TPU:

| GPU                           | TPU                      | How many?                                                                                           |
| :---------------------------- | :----------------------- | :-------------------------------------------------------------------------------------------------- |
| SM (streaming multiprocessor) | Tensor Core              | H100 has 132, TPU has 1-2.                                                                          |
| Warp scheduler                | VPU                      | Each SM has 4, so 132 * 4 = 528 on H100. TPU v5 effectively has 4 per Tensor Core, so 8 total.      |
| SMEM (L1 cache)               | VMEM                     | GPU has 256kB / SM, so ~32MB total. TPUs have around 120MB total of VMEMs at even higher bandwidth. |
| Registers                     | Vector Registers (VRegs) | GPU has 256kB / SM, TPU has 256kB total.                                                            |
| Tensor Core                   | MXU                      | TPU v5p has TPU 4 MXUs per TC, so 8 total. Each H100 SM has 4, so 528 total.                        |

As you can see, TPUs are much less modular than GPUs! This makes them cheaper to build but more complex to program. For instance, TPUs require matmuls to be multiples of a core `[8, 128] x [128, 128]` size and vector work to be done in increments of `[8, 128]`, and will pad input arrays to this size. TPUs are also easy to stall if e.g. vector arithmetic takes longer than a given matrix multiplication, since there is only 1 of each and they are often fused together. But if used properly, they also avoid a huge amount of cost and hardware complexity coming from the modular nature of the GPU.

TPUs also have a lot more fast cache memory that can be used for storing weights and activations. This can make them faster for LLM inference if you can consistently fetch weights into VMEM.

### Worked Problems

Here are some problems to work through that test some of the content above. Answers are provided, but it’s probably a good idea to try to answer the questions before looking, pen and paper in hand.

**Question 1 [CUDA cores]:** How many CUDA cores does an H100 have? How does this compare to the number of independent lanes in a TPU v5p?

{% details Click here for the answer. %}

**Answer:** `132 * 32 * 4 = 16896` CUDA cores. A TPU v5p has 2 TensorCores (usually connected via Megacore), each with a VPU with (8, 128) lanes and 4 independent ALUs per lane, so 2 * 4 * 8 * 128 = 8192. This is half the number of vector lanes of an H100, running at roughly the same frequency.

{% enddetails %}

**Question 2 [Vector FLOPs calculation]:** A single H100 has 132 SMs and runs at a clock speed of 1.59GHz (up to 1.98GHz boost). Assume it can do one vector op per cycle per thread. How many vector FP32 FLOPs can be done per second? With boost? How does this compare to matmul FLOPs?

{% details Click here for the answer. %}

**Answer:** `132 * 32 * 4 * 1.59e9 = 26.9` TFLOPs/s. With boost its 33.5 TFLOPs/s. This is half what’s reported in the [spec sheet](https://www.nvidia.com/en-us/data-center/h100/) because technically we can do an FMA (fused-multiply-add) in one cycle which counts as two FLOPs, but this is basically never achievable. We can do 990 bfloat16 matmul TFLOPs/s, so ignoring FMAs, Tensor Cores do around 30x more FLOPs/s.

{% enddetails %}

**Question 3 [GPU matmul intensity]:** What is the peak bf16 matmul intensity on an H100? A B200?

{% details Click here for the answer. %}

**Answer:** For an H100, we have a peak 990e12 bf16 FLOPs and 3.35e12 bytes / s of bandwidth. So the critical intensity is 990e12 / 3.35e12 = 295, fairly similar to the 240 in a TPU. For B200 its 2250e12 / 8e12 = 281, very similar. This means, similar to TPUs, that we need a batch size of around 280 to be compute-bound in a matmul.

{% enddetails %}

**Question 4 [L1 cache capacity]:** What is the total L1 cache/SMEM capacity for an H100? What about register memory? How does this compare to TPU VMEM capacity.

{% details Click here for the answer. %}

**Answer:** We have 256kB per SM, so about 33MB of each, or about 66MB total. This is about half the 120MB of a modern TPU’s VMEM, although a TPU only has 256kB of register memory total! TPU VMEM latency is lower than SMEM latency, which is one reason we can get away with so little register memory on GPU.

{% enddetails %}

## Networking

Networking is arguably the area where GPUs and TPUs differ the most. As we’ve seen, TPUs are connected in 2D or 3D tori, where each TPU is only connected to its neighbors. This means sending a message between two TPUs must pass through every intervening TPU, and forces us to use only uniform communication patterns over the mesh. While inconvenient in some respects, this also means the number of links per TPU is constant and we can scale to arbitrarily large TPU “pods”.

GPUs on the other hand use a more traditional hierarchical tree-based switching network. Sets of 8 GPUs called **nodes** (up to 72 for B200) are connected within 1 hop of each other with very high bandwidth, and these nodes are connected to larger units (called SUs or scalable units) with a network switch (branded NVSwitch), which in turn are connected into larger units with higher level switches.

### Node Level

A GPU node is a small unit, typically of 8 GPUs (up to 72 for B200), with all-to-all, full-bandwidth connectivity. Each node has some number of NVSwitches, connected to all the local GPUs with high-bandwidth Infiniband NVLinks.

The actual node-level topology has changed quite a bit over time, including the number of switches per node, but for H100, we have 4 NVSwitches per node, with GPUs connected to them in a 5 + 4 + 4 + 5 link pattern:

{% include figure.liquid path="assets/gpu/nvlink-nodes.png" class="img-fluid" caption="<b>Figure:</b> how different NVIDIA GPU generations have connected their GPUs into nodes. Note that the networking configuration and the number of NVSwitches per node has changed from generation to generation." %}

For an H100, each NVLink link has 25GB/s of bandwidth each way (50GB/s for B100), giving us 18 * 25=450GB/s of full-duplex bandwidth from each GPU into the network. These massive switches have up to 64 NVLink ports, meaning for an H100 with 4 switches, they can handle a total of 64 * 25e9 * 4=6.4TB/s of bandwidth.

{% include figure.liquid path="assets/gpu/nvlink4.png" class="img-fluid" caption="<b>Figure:</b> an NVIDIA sales diagram showing how a single NVLink4 Switch works (including 64 ports each with 50GB/s of bandwidth.)" %}

Here’s an overview of how these numbers have changed with GPU generation:

| NVLink Gen | NVSwitch Gen | GPU Generation | NVLink Bandwidth (GB/s, full-duplex) | Max Links / GPU | Node GPU to GPU bandwidth (GB/s full-duplex) | Node size (NVSwitch domain)         |   NVSwitches per node    |
| :--------: | :----------: | :-------------: | :----------------------------------: | :-------------: | :------------------------------------------: | :----------------------------------: | :----------------------: |
|  **3.0**   |   **2.0**    | Ampere         |                  25                  |       12        |                     300                      | 8                                   |            6             |
|  **4.0**   |   **3.0**    | Hopper         |                  25                  |       18        |                     450                      | 8                                   |            4             |
|  **5.0**   |   **4.0**    | Blackwell      |                  50                  |       18        |                     900                      | 8 for H200/B200, 72 for GB200 NVL72 | 2 for B200, 18 for NVL72 |

### Worked Problems

Here are some more Q/A problems on networking. I find these particularly useful to do out, since they make you work through the actual communication patterns.

**Question 1 [Total bandwidth for H100 node]:** How much total bandwidth do we have per node in an 8xH100 node with 4 switches? *Hint:* consider both the NVLink and NVSwitch bandwidth.

{% details Click here for the answer. %}

**Answer:** we have Gen4 4xNVSwitches, each with 64*25e9=1.6TB/s of unidirectional bandwidth. That would give us 4 * 1.6e12=6.4e12 bandwidth at the switch level. However, note that each GPU can only handle 450GB/s of unidirectional bandwidth, so that means we have at most 450e9 * 8 = 3.6TB/s bandwidth. Since this is smaller, the peak bandwidth is 3.6TB/s.

{% enddetails %}

**Question 2 [Bisection bandwidth]:** Bisection bandwidth is defined as the smallest bandwidth available between any even partition of a network. In other words, if split a network into two equal halves, how much bandwidth crosses between the two halves? Can you calculate the bisection bandwidth of an 8x H100 node? *Hint:* bisection bandwidth typically includes flow in both directions.

{% details Click here for the answer. %}

**Answer:** Any even partition will have 4 GPUs in each half, each of which can egress 4 * 450GB/s to the other half. Taking flow in both directions, this gives us 8 * 450GB/s of bytes cross the partition, or 3.6TB/s of bisection bandwidth. This is what NVIDIA reports e.g. [here](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf).

{% enddetails %}

**Question 3 [AllGather cost]:** Given an array of B bytes, how long would a (throughput-bound) AllGather take on an 8xH100 node? Do the math for bf16[DX, F] where D=1024, F=16,384. *It’s worth reading the TPU collectives [section](https://jax-ml.github.io/scaling-book/sharding/) before answering this. Think this through here but we’ll talk much more about collectives next.*

{% details Click here for the answer. %}

**Answer:** Each GPU can egress 450GB/s, and each GPU has $B / N$ bytes (where N=8, the node size). We can imagine each node sending its bytes to each of the other $N - 1$ nodes one after the other, leading to a total of $N - 1$ turns each with $T_\text{comms} = (B / (N * W_\text{unidirectional}))$, or $T_\text{comms} = (N - 1) * B / (N * W_\text{unidirectional})$. This is approximately $B / (N * W_\text{uni})$ or $B$ / 3.6e12, the bisection bandwidth.

For the given array, we have B = `1024*16384*2=32MB`, so the total time is `33.5e6 / 3.6e12 = 9us`. This could be latency-bound, so it may take longer than this in practice.

{% enddetails %}

### Beyond the Node Level

Beyond the node level, the topology of a GPU network is less standard. NVIDIA publishes a reference DGX SuperPod architecture that connects a larger set of GPUs than a single node, but customers and datacenter providers are free to customize this to their needs.

Here is a diagram for a standard 1024 GPU H100 system, where each DGX pod in the bottom row has 8 GPUs. Each set of 32 nodes is called a “Scalable Unit” (or SU), under a single set of 8 leaf switches. This SU has 256 GPUs with 4 NVSwitches per node and 8 leaf switches. The overall SuperPod then adds 16 top level “spine” switches, giving us 1024 GPUs with 512 node-level switches, 32 leaf switches, and 16 spine switches, for a total of 512 + 32 + 16 = 560 NVSwitches. Leaf switches are connected to nodes in sets of 32 nodes, so each set of 256 GPUs has 8 leaf switches. All leaf switches are connected to all spine switches.

At each level we can be bottlenecked by the available NVLink bandwidth, the cabling, or the total switch bandwidth.

  * **Node level:** at the node level, we have 4 * 1.6TB/s = 6.4TB/s of unidirectional switch bandwidth, but each of our 8 GPUs can only egress 450GB/s into the switch, meaning we actually have a peak bandwidth of 450e9 * 8 = 3.6TB/s within the node.
  * **SU/leaf level:** at the SU level, we have 8 switches connecting 32 nodes in an all-to-all fashion with 1x400 Gbps Infiniband. This gives us `8 * 32 * 400 / 8 = 12.8TB/s` of egress bandwidth from the nodes, and we have 8 * 1.6TB/s = 12.8TB/s at the switch level, so both agree precisely.
  * **Spine level:** at the spine level, we have 16 switches connecting 32 leaf switches with 2x400 Gbps links, so we have 32 * 16 * 400 * 2 / 8 = 51.2TB/s of egress bandwidth. The 16 switches give us 16 * 1.6TB/s = 25.6TB/s of bandwidth, so this is the bottleneck at this level.

Per GPU, this gives us 450GB/s of GPU to GPU bandwidth at the node level, 50GB/s at the SU level, and 25 GB/s at the spine level.

| Level     | Number of GPUs | Number of NVSwitches per Unit | Total Bandwidth per Unit (TB/s, full-duplex) | GPU-to-GPU Bandwidth (GB/s, full-duplex) |
| :--------: | :-------------: | :----------------------------: | :-------------------------------------------: | :---------------------------------------: |
| Node      | 8              | 4                             | 3.6                                          | 450                                      |
| Leaf (SU) | 256            | 8                             | 12.8                                         | 50                                       |
| Spine     | 1024           | 16                            | 25.6                                         | 25                                       |

By comparison, a TPU v5p has about 90GB/s egress bandwidth per link, or 540GB/s egress along all axes. This is not point-to-point so it can only be used for restricted, uniform communication patterns but can scale up to 8000 TPUs without loss of bandwidth.

The GPU switching fabric can in theory be extended to arbitrary sizes by adding additional switches or layers of indirection, at the cost of additional latency and reduced bandwidth at the farthest distances.

## How Do Collectives Work on GPUs?

GPUs can perform all the same collectives as TPUs: ReduceScatters, AllGathers, AllReduces, and AllToAlls. Unlike TPUs, the way these work changes depending on whether they’re performed at the node level or above. All these collectives are implemented by NVIDIA in the NCCL (pronounced “nickel”) library, and the actual implementation is a black-box. From here on, we’ll discuss a theoretically optimal model over the NVSwitch tree.

### Within a node

For an AllGather or ReduceScatter at the node level, you can perform them around a ring just like a TPU, using the full GPU-to-GPU bandwidth at each hop. Order the GPUs arbitrarily and send a portion of the array around the ring using the full GPU-to-GPU bandwidth:

$$T_\text{AG or RS comms} = \frac{\text{bytes} \cdot (N - 1)}{N \cdot \text{GPU egress bandwidth}} \rightarrow \frac{\text{bytes}}{\text{GPU egress bandwidth}}$$

For an AllReduce, you can combine an RS + AG as usual for twice the cost. If you’re concerned about latency (e.g. if your array is very small), you can send directly to every GPU in the node at the same time, increasing the total bytes sent by N (the node size) but performing the whole operation in one hop.

So far, this is exactly the same as a TPU, with slightly different overall bandwidth. For instance, with an H100 with 450 GB/s unidirectional bandwidth, AllGather(bf16[B<sub>X</sub>, F]) would take roughly $T_\text{comms} = (2 \cdot B \cdot F) / 450e9$.

#### In network reductions

Since the Hopper generation, NVIDIA switches have supported “SHARP” (*Scalable Hierarchical Aggregation and Reduction Protocol*) which allows for “in-network reductions”. This means the network switches themselves can do reduction operations and multiplex or “MultiCast” the result to multiple target GPUs. This effectively halves the cost of an AllReduce, since it means each GPU can send its data to a top-level switch, have the reduction performed there, and broadcast the result without having to egress each GPU twice.

$$T_\text{AR comms} = \frac{\text{bytes}}{\text{GPU egress bandwidth}}$$

Note: this is exact and not off by a factor of $1 / N$, since each GPU egresses $B * (N - 1) / N$ first, then receives the partially reduced version of its local shard (ingress of $B / N$), finishes the reductions, then egresses $B / N$ again, then ingresses the fully reduced result (ingress of $B * (N - 1) / N$), resulting in exactly bytes $B$ bytes ingressed.

### Beyond the node level

When we go beyond the node-level, the cost is a bit more subtle. We can continue to do a kind of ring reduction, where we think of the ring as being over the leaf or spine switches. For instance, for an AllReduce, we can first AllReduce within the node, then within the leaf, then within the spine, doing a kind of ring reduction each time. In an ideal world, we can overlap these and end up just being bottlenecked by the slowest switch. To a first approximation,

$$T_\text{comms} = \max_i\left(\frac{\text{bytes} \cdot N_{\text{subdomains}_i}}{W_i}\right)$$

where $W_i$ is the aggregate switch bandwidth at level $i$. So for instance, in the above case, we have 3.6TB/s at the node level with 8 subdomains, 12.8TB/s at the SU level with 32 subdomains, and 25.6TB/s at the spine level with 4 subdomains. This means in practice we’ll be bottlenecked by the largest ratio, i.e. `max(8 / 3.6e12, 32 / 12.8e12 , 4 / 25.6e12) = max(2.2e-12, 2.5e-12, 1.56e-13)`, so in practice $T_\text{comms} = B \cdot \text{2.5e-12} = B / 400e9$, i.e. we have about 400GB of AllReduce bandwidth even at the highest level.

In general, the AllReduce bandwidth is $\max_i(N_{\text{subdomains}_i} / W_i)$, so above it is 400GB/s, determined by the leaf level switch. AllGather is a bit more tricky because the actual volume communicated changes at each level. It’s roughly the same but a bit closer to $\max_i(\text{bytes} \cdot (N - 1) / W)$.

### Worked problems

**Question 1 [Single-node AR]:** Consider a single node with N GPUs per node. Precisely how many bytes are ingressed and egressed by the switch during an AllReduce?

{% details Click here for the answer. %}

**Answer:** let’s do this step by step.

1.  Each GPU sends $B \cdot (N - 1) / N$ bytes, so we have $N \cdot B \cdot (N - 1) / N = B \cdot (N - 1)$ ingressed.
2.  We accumulate the partial sums, and we send back $B / N$ bytes to each GPU, so $N \ B / N = B$ bytes egressed.
3.  We do a partial sum on the residuals locally, then send this back to the switch. This is a total of $N * B / N = B$ bytes ingressed.
4.  We capture all the shards and multicast them, sending $B * (N - 1) / N$ to $N$ destinations, for a total of $B * (N - 1) / N * N = B * (N - 1)$ egressed.

Therefore the total is $B * (N - 1) + B = B\cdot N$ bytes ingressed and egressed. This supports the overall throughput being exactly $B\cdot N / W_\text{switch}$.

{% enddetails %}

**Question 2 [SU AllGather]:** Consider only a single SU with M nodes and N GPUs per node. Precisely how many bytes are ingressed and egressed by the node level switch during an AllGather? What about the top-level switch?

{% details Click here for the answer. %}

**Answer:** This is similar to the above, and we’ll do it in stages again.

1.  Each GPU sends $B / MN$ bytes to the switch, for a total ingress of $NB / MN = B / M$ bytes ingress.
2.  We egress the full $B / M$ bytes up to the spine switch.
3.  We ingress $B * (M - 1) / M$ bytes from the spine switch
4.  We egress $B - B / MN$ bytes $N$ times, for a total of $N * (B - B / MN) = NB - B / M$.

The total is $B$ ingress and $BN$ egress, so we should be bottlenecked by egress, and the total time would be 

$$T_\text{AllGather} = \frac{BN}{W_\text{node}} = \frac{B}{450e9}$$.

For the spine switch, the math is actually simpler. We must have B / M bytes ingressed M times (for a total of B bytes), and then B (M - 1) / M egressed M times, for a total of B * (M - 1) out. Since this is significantly larger, the cost is 

$$T_\text{AllGather} = \frac{B \cdot (M - 1)}{W_\text{top}}$$

{% enddetails %}

## Takeaway for LLM Scaling on GPUs

Reproducing the table above for an H100 SuperPod, we have

| Level | Number of GPUs | GPU-to-GPU Bandwidth (full-duplex, GB/s) | Total Bandwidth (full-duplex, TB/s) | AllReduce Bandwidth (GB/s)        |
| :---: | :------------- | :--------------------------------------- | :---------------------------------- | :-------------------------------- |
| Node  | 8              | 450                                      | 3.6 (bounded by NVLink)             | 450GB/s                           |
| Leaf  | 256            | 50                                       | 8 * 1.6 = 12.8                     | 400GB/s                           |
| Spine | 1024           | 25                                       | 16 * 1.6 = 25.6                    | 400GB/s (inherited from the leaf) |

Let’s look at the compute communication rooflines as we did for TPUs. Here, [as before](../training), we’ll compare the computation and communication time and look at what point $T_\text{math} \gt T_\text{comms}$.

**Data parallelism:** To be compute-bound for pure data parallelism within a single node, with in-network reductions, we have

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot DF}{W_\text{AllReduce}}$$

Where we lose a factor of 2 in the comms since we have in-network reductions enabled. Therefore, to be compute-bound, we need $2B / (XC) \gt 1 / W_\text{AllReduce}$ or $B / X \gt C / (2 \cdot W_\text{AllReduce})$, so we just need the per-GPU batch size > `989e12 / (2 * 450e9) = 1098`, quite similar to a TPU (where the number is 850 with all three axes). If we try to do this at the SU or spine level, we get $BS \gt 989e12 / (2 \cdot 400e9) = 1236$.

**FSDP:** For FSDP, which is the only really useful thing, the number is double this, so for FSDP we need BS > 2472 per GPU. Note how much larger this number is than for a TPU (although B100 will reduce this by a factor of 2).

Since we need FSDP of some kind, this means for instance, for 2048 GPUs, we would need a batch size of 5M tokens at a minimum, which is fairly doable.

**Model parallelism:** For model parallelism, this suggests within a single node, we need $Y \< F / (898e12 \cdot (8-1) / (8 \cdot 450e9)) = F / 1746$, so for F=16k, we can go up to 9-way parallelism (or really 8 way, since that’s how large a node is). Clearly, we can’t go larger than this, but not because cross-node bandwidth is low but because our overall bandwidth is low.

**Mixed FSDP + model parallelism:** Combining some form of model parallelism with DP isn’t quite as simple as in a TPU mesh, where we reduce the cost of the AllReduce by Y (the amount of TP). The general rule for a tree $\text{AllReduce}_X(A_Y { U_X })$ (assuming Y is the inner axis) seems to be

$$T_\text{comms} = \max_i\left[\frac{B \cdot S_i}{\max(Y, S_{i-1}) \cdot W_i}\right]$$

where $S_i$ is M * N * …, the size of the subnodes at level $i$ in the tree. So if Y is 64, then at the node level we have `B * 8 / (64 * 3.6e12)`, at the leaf level we have `B * 256 / (64 * 12.8e12)`, and at the spine level we have B * 1024 / (256 * 25.6e12), or in other words bandwidths of 28.8e12, 3.2e12, and 6.4e12. This means we effectively gain a factor of 64 at the node level, a factor of 8 in bandwidth at the leaf level, and stay constant at the spine level. If we did 256-way expert sharding, we’d get a 256x speedup at the node level, 32x at the leaf level, and no speedup at the spine level.

Therefore, if we do 8-way model parallelism, we do in fact reduce the cost of the node-level reduction by 8 and leave everything else the same, so it’s more or less free but not useful in reducing the overall cost of the reduction.

**Expert parallelism:** Expert parallelism isn’t discussed in much detail in the main body of this book, but to a first approximation, an MoE replicates the MLP weights of a dense model E times, i.e. turning $W_\text{in}[D, F]$ and $W_\text{out}[F, D]$ into $W_\text{in}[E, D, F]$ and $W_\text{out}[E, F, D]$, but each token only activates k of these. This increases the total memory by E but increases the FLOPs by only k times. This adds two AllToAlls to send tokens to their chosen experts, but also requires us to AllReduce E times more bytes in a DP setup.

AllToAlls are simpler in a GPU than a TPU because they can be done directly point-to-point, so in this sense the story is much cleaner.

But the extra memory is a bigger issue. On a TPU, we can simply shard the experts along a completely independent axis, e.g. $W[E_z, D_X, F_Y]$, which reduces us to 2 axes of a more or less standard dense model. But on GPUs, we can no longer shard our experts along a separate axis, so we can no longer totally alleviate the cost of the extra communication. As we noted above, by doing more sharding of any kind, but particularly expert parallelism, we can reduce the cost of the AllReduce somewhat. If we did 8-way expert parallelism at the node level, we reduce the node-level cost but not the spine level cost, which is bad. If we do 64-way or even 256-way, we get significant wins, but the roofline is no longer quite so simple, since for DP=X and EP=Y, with k tokens per expert, we have (in the backward pass)

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot k \cdot B \cdot D \cdot F}{X \cdot Y \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot E \cdot D \cdot F}{W_\text{AllReduce}}$$

Where $W_\text{AllReduce}$ will be 8-times higher at the leaf level. This still gives us

$$\frac{B}{X} \gt \frac{E \cdot C}{2 \cdot k \cdot W_\text{AllReduce}}$$

So with 8-way EP, we don’t really benefit at the leaf level. Taking DeepSeek with 256 experts and 8 activated (roughly) with 64-way EP (ignoring pipeline parallelism), suddenly the cost drops to `1236 * 256 / 8 / 8 = 4944`, meaning we would need `4944 * 2048 = 10M` tokens, which is actually doable. This is ignoring the extra factor of two from FSDP, which would bring this up to 20M.

**Pipeline parallelism:** Pipeline parallelism has a very minor cost since we are just hopping a small message (the activations or activation gradients) over the top-level switch. This counts as an extra form of model parallelism, although it makes FSDP more challenging.

**What does DeepSeek do?** For reference, DeepSeek is trained with 2048 H800 GPUs with:

  * 64-way Expert Parallelism (EP) spanning 8 nodes
  * 16-way Pipeline Parallelism (PP)
  * 2-way ZeRO-1 Data Parallelism (DP)

They had a steady state batch size of `4096 * 15360 = 62,914,560` tokens. You can see that with 64-way EP and 16-way PP, we end up with 1024-way model parallelism in total, which means the AllReduce is done at the spine level. This gives us a lot more bandwidth from the fat tree to work with, so we have no issue with more data parallelism. We could do as little as 4-way pipeline parallelism and still be in this situation, although that has its own issues (bubbles).

**TLDR of GPU scaling:**

* Pure data parallelism is amazing because SHARP reduces AllReduce cost, but not very useful for big models.
* FSDP is fine, 2x cost and roofline of pure DP because we have to do an AG + RS or AR + AG (the second is better for pipelining). ZeRO-1 works with pipelining, ZeRO-3 doesn’t. \~2k tokens per GPU needed.
* MP + FSDP is fine but not great, model parallelism can’t scale beyond a node for pure bandwidth reasons (nothing to do with cross-node bandwidth being less). Better in B100/B200. Reduces memory per GPU but doesn’t help otherwise, i.e. doesn’t reduce critical batch size because it doesn’t reduce leaf-level bandwidth.
* In general, MoEs + DP is a bit harder because the experts are so chunky, so DP/FSDP rooflines go way up unless we do a lot of expert parallelism. Really need expert parallelism beyond the node level, like ideally EP + MP takes up an entire SU, so e.g. 8-way model parallelism, 32-way expert parallelism works well. Need to span many nodes to reduce leaf-level bandwidth.
* Pipeline parallelism works fine if you can handle the code complexity of zero-bubble pipelining. Makes ZeRO-3 impossible, so have to do ZeRO-1 instead. Counts as model parallelism.
* **Main TLDRs:**
    * For smallish dense models, 8-way TP + pure data parallelism would be very strong. For H100 you could go up to a 64B model in bf16.
    * For larger dense models, can do TP + PP or FSDP depending on batch size. More PP lets you go to a smaller batch size, but FSDP is fine in most cases.
    * For MoEs, can do some combination of TP + EP + PP that gets you up to the spine level, then you’re fine because you have tons of bandwidth at the spine level with the fat tree. Can’t go past node-level TP, EP bounded by the number of experts and the cost of imbalance + A2A, PP bounded by microbatch size. Then do pure DP or ZeRO-1/3 beyond that.

### Worked problems

**Question 1 [Cross-node AR cost]:** Consider an array bf16[D<sub>X</sub>, F<sub>Y</sub>] sharded over a single node of N GPUs. How long does $\text{AllReduce}(bf16[D, F_Y] { U_X })$ take? You can assume we do in-network reductions. Explain how this differs if we have more than a single node?

{% details Click here for the answer. %}

**Answer:** we can try to modify the answer to the similar question above. Basically, we first egress $B * (X - 1) / XY$ bytes from each GPU, then send back $B / XY$ to each GPU, then send that same amount back to the switch, then send $B * (X - 1) / XY$ back to each GPU. The total is $NB / Y$ ingress and egress, so the total time is $T_\text{comms} = NB / (Y \cdot W_\text{switch}) = N \cdot 2DF / (\left Y\rvert \cdot W_\text{switch})$, so the total time does decrease with Y.

If we go beyond a single node, we can do roughly the same reduction as above, but when we egress the node-level switch, we need to send all B bytes, not just B / Y. This is because we need to keep each model shard separate.

{% enddetails %}

**Question 2 [Spine level AR cost]:** Consider the same setting as above, but with 256-way model parallelism (so the AR happens at the spine level). How long does the AllReduce take? What batch size per GPU could we handle here?

{% details Click here for the answer. %}

**Answer:** This lets us take advantage of the rather ludicrous amount of bandwidth at the spine level. We have 25.6TB/s of bandwidth over 4 nodes, so an AllReduce bandwidth of 6.4TB/s. Using SHARP, this could take as little as $2 \cdot D \cdot F / 6.4e12$ seconds.

This means in theory we can have as small a batch size as `989e12 / (2 * 6.4e12) = 77` tokens per GPU, or 19,712 per SU, which is pretty wild. This could be the case if we did something like 8-way model parallelism within the node, and 32-way expert parallelism across nodes, or some form of pipelining.

{% enddetails %}

## How does this change with B100 and the GB200 NVL72?

Broadwell introduces a bunch of major networking changes, including NVLink 5 with twice the overall bandwidth (900GB/s) and much larger nodes (72 GPUs in NVL72). Here's a diagram:

{% include figure.liquid path="assets/gpu/b100-node.png" class="img-fluid" caption="<b>Figure:</b> a diagram showing how a B100/B200 NVL72 node is constructed, with 18 switches and 72 GPUs." %}

The first order effect is that all our rooflines get roughly twice as good: all our AllReduces and AllGathers are twice as fast, so we can do twice as much of them. The BS > 1098 bound we calculated for pure data parallelism decreases to 549, which is close to a TPU v5p. The model parallelism bound for F = 16000 increases to 18, meaning we can do nearly twice the amount of model parallelism.

NVIDIA also has plans to build a 576 GPU GB200 NVL576 topology that has two layers of switches but can achieve full bandwidth between all GPUs. This is roughly a node although it will have some minor added latency between more distant GPUs. This has not yet been launched.

{% include figure.liquid path="assets/gpu/nvl-576.png" class="img-small" caption="<b>Figure:</b> a diagram showing how we could see 576 GPU nodes in Broadwell." %}
