# LocalAI

LocalAI is a local-first, private personal AI assistant that runs entirely on-device on small language models. It is designed as a lightweight alternative to cloud-hosted assistants, with no data leaving the machine, no subscription, and full user control.

The core thesis is that SLMs are sufficient for personal assistance when the orchestration around them is strong. Models are treated as the fixed, constrained ingredient and the engineering around them compensates for limitations in tool use, context window, and multi-step reasoning.

The project is intentionally local and self-hosted, starting with quantized models via Ollama on desktop and designed to extend to low-end and mobile runtimes via llama.cpp and ONNX Runtime.
