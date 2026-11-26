# How it works

1. **Audio in:** a WAV is converted to 16 kHz mono. Audio over 20 seconds is cut at its quietest points.
2. **Features:** each piece becomes a log-mel spectrogram, 128 bands every 10 ms.
3. **Encoder:** a 24-layer FastConformer turns the spectrogram into one vector per 80 ms. This is most of the
   work: on the processor it runs int8 matrix kernels (AVX-VNNI-INT8, AVX-VNNI or AVX2, the fastest the CPU
   has, and plain 128-bit vectors on CPUs without AVX2) with one thread per fast core; on a GPU it runs as
   Direct3D 12 compute shaders.
4. **Decoder:** TDT greedy decoding picks the next word piece, and how many frames to skip, until the audio
   ends. The pieces become text with punctuation and capitals.

The large matrices stay 8-bit in memory, as in the GGUF file: int8 values with one scale per block of 32, on
the processor or the graphics card. Norms, biases and the small tables are floating point.

## In more detail

- [model-and-decoding.md](model-and-decoding.md): GGUF reading, Q8_0 blocks, the mel front end, encoder,
  TDT decoding, model install
- [cpu-kernels.md](cpu-kernels.md): the int8 CPU paths, quantization, threads and the thread budget
- [gpu-path.md](gpu-path.md): the Direct3D 12 host side: adapters, pipelines, buffers, the Q8 GEMM tiles
- [gpu-shaders.md](gpu-shaders.md): what every shader entry point computes
- [microphone.md](microphone.md): WASAPI capture into memory
- [benchmarks-howto.md](benchmarks-howto.md) and [benchmarks.md](benchmarks.md): how to measure, and every
  measurement so far
- [ideas.md](ideas.md): performance ideas not built yet
