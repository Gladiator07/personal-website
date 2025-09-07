---
title: "My Second Blog Post"
description: "A brief description of what this post is about"
date: "2025-09-07"
categories: [LLMs]
image: ../profile.jpg
---

I was just watching a Sentdex video and found out this pretty cool example on where we could use Pipeline Parallelism on his Jetson Thor which has a slow bandwith typically compared to a production GPU. This is a really good example of it. Here’s what was the moat of that approach:

- **Bottleneck:** The Jetson Thor’s low memory bandwidth (273 GB/s) caused per-request latency (~500 ms) in the MoonDream 2 VLM, limiting throughput to about 2 FPS. It wasn’t compute or networking; “we know the bottleneck is the memory bandwidth.”
- **Fix (pipeline-style parallelism):** He exploited the large 128 GB memory by running many independent VLM servers in parallel (each ~5 GB). Frames were interleaved across servers (1,3,5… to server A; 2,4,6… to server B; etc.). This keeps single-frame latency the same but increases overall frame throughput.
- **Result:** Doubling servers doubled FPS; 10 servers ≈ 20 FPS; 15 servers ≈ 30 FPS, with high GPU utilization (often 90%+). Trade-off: fewer headroom for other models, but you can tune server count.

