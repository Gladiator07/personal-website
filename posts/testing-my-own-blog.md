---
title: "vLLM V0 + Prefix Caching on 2×T4: Why merged weights hung but LoRA didn't"
description: "A brief description of what this post is about"
date: "2025-09-07"
categories: [new]
image: ../profile.jpg
---

# vLLM V0 + Prefix Caching on 2×T4: Why merged weights hung but LoRA didn't

---

### TL;DR (Too long didn't read)
* Your freeze was a **V0 engine + TP>1 + prefix caching** interaction under tight VRAM. Turning **enable_prefix_caching=False** removed that path.
* With **merged weights**, the engine **reused** one giant shared-prefix KV cache across thousands of requests; the V0 scheduler/block-manager then ran out of allocatable KV blocks and stalled. With **LoRA**, caches are **namespaced per adapter**, so that aggressive sharing path wasn't taken, avoiding the stall.

---

### What prefix caching is doing in vLLM
* Prefix caching stores **KV-cache blocks** for a shared prompt prefix so later requests can **skip prefill** for those tokens; it's implemented via **block hashing + LRU** over KV blocks.
* In V0/V1 design notes and RFCs, cached KV blocks are treated like an OS cache: blocks are **retained** instead of freed and tracked in a global structure for reuse.

---

### Why merged weights + prefix caching froze on your run
1. **Scheduler blind spot (V0 path):**

   In V0, the scheduler decides whether it **can allocate** all KV blocks for a sequence **before** it accounts for which blocks are already cached. So even when most of your long prefix is cacheable, the scheduler may still demand a large number of *free* blocks up front. With thousands of identical prompts, this starves the block manager and the run stalls.
2. **Tight VRAM → no headroom:**

   vLLM **pre-allocates** KV blocks up to a fraction of GPU memory (gpu_memory_utilization). With 16 GiB T4s, TP>1, long context, and high utilization, the pool has little slack; coupled with (1), you hit a deadlock/starvation pattern (first request OK, then nothing). Lowering utilization or disabling prefix caching frees the pipeline.
3. **This exact failure mode is known:**

   Multiple reports show hangs/crashes when **prefix caching is enabled**, where *"first request is fine, second returns nothing"*. Disabling prefix caching or reducing memory pressure is the workaround.

---

### Why LoRA didn't show the problem (but merged weights did)
* With **multiple adapters**, vLLM treats KV caches as **separate per LoRA** (cache keys include the adapter identity), so sequences using an adapter don't reuse the base-model prefix cache. That **reduces cache sharing pressure** and avoids the problematic allocator/scheduler path on V0. After you **merged** LoRA → base, there's **one** weight set again, so aggressive prefix reuse kicked in and exposed the stall.

---

### What to do on Kaggle 2×T4 (V0 engine)
* Keep **enable_prefix_caching=False** when you have a long, identical prompt and **TP>1**.
* Leave **VRAM headroom**: gpu_memory_utilization≈0.80, trim max_model_len to what you actually need, and **batch in small chunks** (32–128 prompts per generate). These align with vLLM's guidance to avoid KV-block starvation.

---

### Notes on V1 vs V0 (why you read conflicting advice)
* V1 reworked prefix caching data structures; it's near-zero overhead and on by default in many builds, but you're on **T4 → V0 fallback**, so you still hit the older scheduler/block-manager behavior. Some V1 docs even suggest toggling prefix caching/eager mode depending on model/backend. The takeaway: **your hardware keeps you on V0**, so use the V0-safe settings above.

---

### One-line mental model

**Merged weights** made all requests share one hot prefix cache → **V0 scheduler demanded more free KV blocks than available** → **starvation**. **LoRA** segmented caches by adapter → less reuse → no starvation.   