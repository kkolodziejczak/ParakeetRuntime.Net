# Third-party notices

## NeMo-Speech.cpp (Apache-2.0)

ParakeetRuntime's engine is a C# port of the parts of [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) it needs: the log-mel front end, the FastConformer encoder, TDT greedy decoding and punctuation spacing. No NeMo-Speech.cpp source or binary is included.

> NeMo-Speech.cpp
> Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
>
> Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Parakeet TDT 0.6B v3 model (CC BY 4.0)

ParakeetRuntime does not include a speech model. It runs NVIDIA's [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) in Q8_0 GGUF form, which you download separately. The model is © NVIDIA and licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
