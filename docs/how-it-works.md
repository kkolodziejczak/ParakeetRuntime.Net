# How it works

ParakeetRuntime.Net turns audio into punctuated text in four stages, all inside your process:

1. WAV audio is converted to 16 kHz mono. Audio longer than 20 seconds is cut at its quietest points into pieces of 20 seconds or less.
2. Each piece becomes a log-mel spectrogram with 128 bands every 10 ms.
3. A 24-layer FastConformer encoder turns the spectrogram into one vector per 80 ms. This is where most of the compute time goes.
4. A TDT greedy decoder picks the next word piece and how many frames to skip until the audio ends, producing text with punctuation and capitalization.

On a GPU, the entire pipeline from the log-mel spectrogram through subsampling, the FastConformer encoder, the joint projection, and the TDT greedy decoder runs as Direct3D 12 compute shaders, keeping intermediate data in VRAM. On the CPU, the heavy matrix multiplications run through int8 kernels (`AVX-VNNI-INT8`, `AVX-VNNI`, or `AVX2`, picking the fastest your CPU supports, or 128-bit vectors on CPUs without AVX2) across one thread per fast core.

On both CPU and GPU, the large weight matrices stay 8-bit in memory as they are stored in the GGUF file: `int8` values with one scale per block of 32. Norms, biases, and small tables stay floating point.

## In more detail

- [model-and-decoding.md](model-and-decoding.md) goes through GGUF reading, Q8_0 blocks, the mel front end, encoder, TDT decoding, and model installation.
- [cpu-kernels.md](cpu-kernels.md) explains the int8 CPU paths, quantization, threads, and the thread budget.
- [gpu-path.md](gpu-path.md) describes the Direct3D 12 host side: adapters, pipelines, buffers, and Q8 GEMM tiles.
- [gpu-shaders.md](gpu-shaders.md) lists what every compute shader entry point computes.
- [microphone.md](microphone.md) covers WASAPI capture into memory.
- [benchmarks-howto.md](benchmarks-howto.md) and [benchmarks.md](benchmarks.md) show how to measure performance and record every run so far.
- [ideas.md](ideas.md) tracks performance ideas that are not built yet.
