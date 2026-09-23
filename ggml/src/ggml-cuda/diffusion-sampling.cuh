#pragma once

#include "common.cuh"

bool ggml_cuda_diffusion_sample_topk(
        ggml_backend_t backend,
        const ggml_tensor * logits,
        const ggml_cuda_diffusion_sample_params * params,
        ggml_cuda_diffusion_sample_result * result);

bool ggml_cuda_diffusion_read_logprobs(
        ggml_backend_t backend,
        const ggml_tensor * logits,
        int32_t n_vocab,
        int32_t n_tokens,
        const int32_t * requested_ids,
        int32_t n_requested_ids,
        float * out_logprobs);
