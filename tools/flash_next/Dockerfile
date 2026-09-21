# Preservation rebuild recipe. No GPU or model access during compilation.
# GPU validation of a freshly rebuilt image is still required before serving.
FROM docker.io/vllm/vllm-openai@sha256:43f13b4c624ab9e9e6753d0eeb5953268bff334f2a826239e6f4a2197d47bb96

COPY engine/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY builder/ /tmp/flash-builder/
COPY csrc/ /tmp/source/csrc/

ENV TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2
RUN mkdir /tmp/build \
    && uv venv --system-site-packages --python /usr/local/bin/python3 /tmp/flash-builder/.venv \
    && /tmp/flash-builder/.venv/bin/python /tmp/flash-builder/build_flash_marlin_schedule.py \
       --source /tmp/source/csrc --output /tmp/build \
    && echo '78541fcf5b9f9fa3ce3bc5f7fcb8de7f558c0f58be91def817a864bf3d040912  /tmp/build/flash_marlin_dp4/flash_marlin_dp4.so' | sha256sum -c - \
    && cp /tmp/build/flash_marlin_dp4/flash_marlin_dp4.so \
       /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/flash_marlin_dp4.so

# Experimental K32 remains disabled unless its exact runtime gate is enabled.
RUN mkdir /tmp/build-k32 \
    && /tmp/flash-builder/.venv/bin/python /tmp/flash-builder/build_flash_marlin_k32.py \
       --source /tmp/source/csrc --output /tmp/build-k32 \
    && echo '1d488454923f7f9b80f60172513f286371a0ac40671ea0dacac3f00aaa897a61  /tmp/build-k32/flash_marlin_k32/flash_marlin_k32.so' | sha256sum -c - \
    && cp /tmp/build-k32/flash_marlin_k32/flash_marlin_k32.so \
       /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/flash_marlin_k32.so

# The engine's opt-ins remain default-off. The documented profile selects them
# only after image qualification. Keep the parent's standard vLLM entrypoint.
