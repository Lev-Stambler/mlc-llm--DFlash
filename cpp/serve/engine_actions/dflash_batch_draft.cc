/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/dflash_batch_draft.cc
 * \brief DFlash batch draft action: single-pass block diffusion draft.
 *  Unlike EAGLE's iterative drafting, DFlash generates all block_size tokens
 *  in a single forward pass through the lightweight draft model.
 */

#include <tvm/runtime/device_api.h>

#include <cmath>
#include <numeric>

#include "../config.h"
#include "../model.h"
#include "../sampler/sampler.h"
#include "action.h"
#include "action_commons.h"
#include "dflash_commons.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The action that runs single-pass block draft proposal for DFlash.
 *  Model 0 = target (provides embed + lm_head), Model 1 = DFlash draft (stateless).
 */
class DFlashBatchDraftActionObj : public EngineActionObj {
 public:
  explicit DFlashBatchDraftActionObj(Array<Model> models, LogitProcessor logit_processor,
                                     Sampler sampler, std::vector<ModelWorkspace> model_workspaces,
                                     DraftTokenWorkspaceManager draft_token_workspace_manager,
                                     EngineConfig engine_config,
                                     Optional<EventTraceRecorder> trace_recorder)
      : models_(std::move(models)),
        logit_processor_(std::move(logit_processor)),
        sampler_(std::move(sampler)),
        model_workspaces_(std::move(model_workspaces)),
        draft_token_workspace_manager_(std::move(draft_token_workspace_manager)),
        engine_config_(std::move(engine_config)),
        trace_recorder_(std::move(trace_recorder)) {}

  Array<Request> Step(EngineState estate) final {
    // - Only run when there are two models and running requests.
    if (models_.size() != 2 || estate->running_queue.empty()) {
      return {};
    }

    // Preempt if needed — only check target model (model 0) pages.
    // DFlash draft model has no KV cache pages.
    std::vector<RequestStateEntry> running_rsentries = estate->GetRunningRequestStateEntries();
    while (!CanDecode(running_rsentries.size())) {
      if (estate->prefix_cache->TryFreeMemory()) continue;
      RequestStateEntry preempted = PreemptLastRunningRequestStateEntry(
          estate, models_, draft_token_workspace_manager_, trace_recorder_);
      if (preempted.same_as(running_rsentries.back())) {
        running_rsentries.pop_back();
      }
    }

    auto tstart = std::chrono::high_resolution_clock::now();

    int num_rsentries = running_rsentries.size();
    TVM_FFI_ICHECK_GT(num_rsentries, 0);

    Array<String> request_ids;
    std::vector<int64_t> request_internal_ids;
    Array<GenerationConfig> generation_cfg;
    std::vector<RandomGenerator*> rngs;
    Array<RequestModelState> mstates;
    request_ids.reserve(num_rsentries);
    request_internal_ids.reserve(num_rsentries);
    generation_cfg.reserve(num_rsentries);
    mstates.reserve(num_rsentries);
    for (const RequestStateEntry& rsentry : running_rsentries) {
      request_ids.push_back(rsentry->request->id);
      request_internal_ids.push_back(rsentry->mstates[0]->internal_id);
      generation_cfg.push_back(rsentry->request->generation_cfg);
      rngs.push_back(&rsentry->rng);
      // Use draft model's mstate (model 1) for draft token tracking,
      // consistent with verify which reads from mstates[draft_model_id_].
      mstates.push_back(rsentry->mstates[1]);
    }

    int block_size = estate->spec_draft_length;
    TVM_FFI_ICHECK_GT(block_size, 0) << "DFlash spec_draft_length (block_size) must be positive.";

    // DFlash single-pass draft: process each request independently through the
    // stateless draft model.
    {
      for (int req_idx = 0; req_idx < num_rsentries; ++req_idx) {
        auto tdraft_start = std::chrono::high_resolution_clock::now();
        int64_t internal_id = request_internal_ids[req_idx];

        // Check that projected_target_hidden exists for this request.
        auto& pth_map = *model_workspaces_[0].dflash_projected_target_hidden;
        auto it = pth_map.find(internal_id);
        if (it == pth_map.end()) {
          continue;
        }
        Tensor projected_target_hidden = it->second.View();

        // TODO(dflash): Read mask_token_id from DFlash model config.
        int mask_token_id = 151669;  // Default for Qwen3-8B-DFlash-b16

        // Build input tokens: first token is the committed token, rest are mask tokens.
        int first_token;
        if (!mstates[req_idx]->draft_output_tokens.empty()) {
          first_token = mstates[req_idx]->draft_output_tokens.back().GetTokenId();
        } else {
          TVM_FFI_ICHECK(!mstates[req_idx]->committed_tokens.empty());
          first_token = mstates[req_idx]->committed_tokens.back().GetTokenId();
        }
        std::vector<int> block_tokens(block_size, mask_token_id);
        block_tokens[0] = first_token;

        // 1. Embed via target model.
        RECORD_EVENT(trace_recorder_, request_ids, "start dflash draft embedding");
        ObjectRef noise_embedding =
            models_[0]->TokenEmbed({IntTuple{block_tokens.begin(), block_tokens.end()}});
        RECORD_EVENT(trace_recorder_, request_ids, "finish dflash draft embedding");

        // Reshape embedding from [block_size, hidden_size] to [1, block_size, hidden_size].
        Tensor embed_nd = Downcast<Tensor>(noise_embedding);
        int hidden_size = embed_nd->shape[1];
        ObjectRef reshaped_embedding =
            embed_nd.CreateView({1, block_size, hidden_size}, embed_nd->dtype);

        // 2. Determine context length from projected_target_hidden shape.
        int ctx_len = projected_target_hidden->shape[1];
        int total_len = ctx_len + block_size;

        // 3. Construct RoPE cos/sin and attention mask.
        const DFlashDraftAuxTensors& aux_tensors =
            GetDraftAuxTensors(total_len, projected_target_hidden->device);

        // 4. DFlash draft forward (single-pass through stateless draft model).
        RECORD_EVENT(trace_recorder_, request_ids, "start dflash draft forward");
        ObjectRef draft_hidden = models_[1]->DFlashDraftForward(
            reshaped_embedding, projected_target_hidden, aux_tensors.cos, aux_tensors.sin,
            aux_tensors.mask);
        RECORD_EVENT(trace_recorder_, request_ids, "finish dflash draft forward");

        // 5. Get logits via target model's LM head (shared).
        // Reshape draft_hidden from [1, block_size, h] to [block_size, h] for GetLogits.
        Tensor draft_hidden_nd = Downcast<Tensor>(draft_hidden);
        Tensor reshaped_hidden =
            draft_hidden_nd.CreateView({block_size, hidden_size}, draft_hidden_nd->dtype);
        Tensor logits;
        bool use_draft_logits = models_[1]->CanGetLogits();
        if (use_draft_logits) {
          logits = models_[1]->GetLogits(reshaped_hidden);
        } else {
          logits = models_[0]->GetLogits(reshaped_hidden);
        }
        // Handle possible 3D output [1, block_size, v].
        if (logits->ndim == 3) {
          logits = logits.CreateView({logits->shape[1], logits->shape[2]}, logits->dtype);
        }
        TVM_FFI_ICHECK_EQ(logits->ndim, 2);
        TVM_FFI_ICHECK_EQ(logits->shape[0], block_size);

        // 6. Apply logit processing for all block_size positions at once.
        // Replicate config for each position (all belong to same request).
        Array<String> rep_request_ids;
        Array<GenerationConfig> rep_gen_cfg;
        Array<RequestModelState> rep_mstates;
        rep_request_ids.reserve(block_size);
        rep_gen_cfg.reserve(block_size);
        rep_mstates.reserve(block_size);
        for (int d = 0; d < block_size; ++d) {
          rep_request_ids.push_back(request_ids[req_idx]);
          rep_gen_cfg.push_back(generation_cfg[req_idx]);
          rep_mstates.push_back(mstates[req_idx]);
        }
        // DFlash block diffusion generates all positions simultaneously — no sequential
        // draft token penalty needed. Pass nullptr for draft_mstates/draft_token_indices.
        logit_processor_->InplaceUpdateLogits(logits, rep_gen_cfg, rep_mstates, rep_request_ids);
        // 7. Sample from positions 1..block_size-1 (skip position 0 = committed token).
        int num_draft = block_size - 1;
        bool deterministic_generation = IsDeterministicGenerationConfig(generation_cfg[req_idx]);
        Tensor probs_on_device{nullptr};
        std::vector<SampleResult> sample_results;
        if (deterministic_generation) {
          Tensor logits_on_host = CopyTensorToCPU(logits);
          sample_results.reserve(num_draft);
          for (int d = 0; d < num_draft; ++d) {
            sample_results.push_back(
                SampleResult{{ArgmaxTokenId(logits_on_host, d + 1), 1.0f}, {}});
          }
        } else {
          probs_on_device = logit_processor_->ComputeProbsFromLogits(logits, rep_gen_cfg,
                                                                     rep_request_ids);
          std::vector<int> sample_indices;
          Array<String> sample_request_ids;
          Array<GenerationConfig> sample_gen_cfg;
          std::vector<RandomGenerator*> sample_rngs;
          sample_indices.reserve(num_draft);
          sample_request_ids.reserve(num_draft);
          sample_gen_cfg.reserve(num_draft);
          sample_rngs.reserve(num_draft);
          for (int d = 0; d < num_draft; ++d) {
            sample_indices.push_back(d + 1);  // Row d+1 in probs tensor
            sample_request_ids.push_back(request_ids[req_idx]);
            sample_gen_cfg.push_back(generation_cfg[req_idx]);
            sample_rngs.push_back(rngs[req_idx]);
          }

          Tensor renormalized_probs = sampler_->BatchRenormalizeProbsByTopP(
              probs_on_device, sample_indices, sample_request_ids, sample_gen_cfg);
          sample_results = sampler_->BatchSampleTokensWithProbAfterTopP(
              renormalized_probs, sample_indices, sample_request_ids, sample_gen_cfg,
              sample_rngs);
          TVM_FFI_ICHECK_EQ(static_cast<int>(sample_results.size()), num_draft);
        }

        // 8. Store draft probs and add draft tokens.
        // Allocate block_size slots so ScatterDraftProbs can scatter all rows.
        // Then free slot 0 (committed token position, not used as draft).
        draft_token_slots_.clear();
        draft_token_workspace_manager_->AllocSlots(block_size, &draft_token_slots_);
        if (!deterministic_generation) {
          models_[0]->ScatterDraftProbs(probs_on_device, draft_token_slots_,
                                         &model_workspaces_[0].draft_probs_storage);
        }
        // Free unused slot for position 0 (committed token's logit row).
        std::vector<int> unused_slots = {draft_token_slots_[0]};
        draft_token_workspace_manager_->FreeSlots(unused_slots);

        // Add block_size-1 draft tokens as a linear chain.
        for (int d = 0; d < num_draft; ++d) {
          int64_t parent_idx = d - 1;  // -1 for first draft, 0 for second, etc.
          mstates[req_idx]->AddDraftToken(sample_results[d], draft_token_slots_[d + 1],
                                          parent_idx);
        }

        auto tdraft_end = std::chrono::high_resolution_clock::now();
        estate->metrics.UpdateDraftTimeByBatchSize(
            1, static_cast<double>((tdraft_end - tdraft_start).count()) / 1e9);
      }
    }

    auto tend = std::chrono::high_resolution_clock::now();
    estate->metrics.engine_decode_time_sum += static_cast<double>((tend - tstart).count()) / 1e9;

    return {};
  }

 private:
  bool CanDecode(int num_rsentries) {
    // Only check target model (model 0) — DFlash draft model has no KV cache.
    int num_available_pages = models_[0]->GetNumAvailablePages();
    return num_rsentries <= num_available_pages;
  }

  DFlashDraftAuxTensors CreateDraftAuxTensors(int total_len, int block_size, DLDevice device) {
    // TODO(dflash): Read head_dim and rope_theta from model config dynamically.
    int head_dim = 128;           // Default for Qwen3-8B-DFlash
    double rope_theta = 1000000.0;  // Default for Qwen3

    int half_dim = head_dim / 2;
    int total_elements = total_len * head_dim;

    std::vector<float> cos_data(total_elements);
    std::vector<float> sin_data(total_elements);

    for (int p = 0; p < total_len; ++p) {
      for (int i = 0; i < half_dim; ++i) {
        double inv_freq = 1.0 / std::pow(rope_theta, 2.0 * i / head_dim);
        double angle = p * inv_freq;
        float cos_val = static_cast<float>(std::cos(angle));
        float sin_val = static_cast<float>(std::sin(angle));
        // Duplicate: positions i and i+half_dim have the same cos/sin.
        cos_data[p * head_dim + i] = cos_val;
        cos_data[p * head_dim + i + half_dim] = cos_val;
        sin_data[p * head_dim + i] = sin_val;
        sin_data[p * head_dim + i + half_dim] = sin_val;
      }
    }

    Shape shape{1, total_len, 1, head_dim};
    Tensor cos = Tensor::Empty(shape, DataType::Float(32), device);
    Tensor sin = Tensor::Empty(shape, DataType::Float(32), device);
    cos.CopyFromBytes(cos_data.data(), cos_data.size() * sizeof(float));
    sin.CopyFromBytes(sin_data.data(), sin_data.size() * sizeof(float));

    Shape mask_shape({1, 1, block_size, total_len});
    Tensor mask = Tensor::Empty(mask_shape, DataType::Float(32), device);
    size_t num_bytes = block_size * total_len * sizeof(float);
    std::vector<char> zeros(num_bytes, 0);
    mask.CopyFromBytes(zeros.data(), num_bytes);
    // Ensure async host-to-device copies complete before local buffers are freed.
    tvm::runtime::DeviceAPI::Get(device)->StreamSync(device, nullptr);
    return {cos, sin, mask};
  }

  const DFlashDraftAuxTensors& GetDraftAuxTensors(int total_len, DLDevice device) {
    auto& aux_cache = *model_workspaces_[0].dflash_draft_aux_tensors;
    auto it = aux_cache.find(total_len);
    if (it == aux_cache.end()) {
      it = aux_cache
               .emplace(total_len,
                        CreateDraftAuxTensors(total_len, engine_config_->spec_draft_length, device))
               .first;
    }
    return it->second;
  }

  Array<Model> models_;
  LogitProcessor logit_processor_;
  Sampler sampler_;
  std::vector<ModelWorkspace> model_workspaces_;
  DraftTokenWorkspaceManager draft_token_workspace_manager_;
  EngineConfig engine_config_;
  Optional<EventTraceRecorder> trace_recorder_;
  std::vector<int> draft_token_slots_;
};

EngineAction EngineAction::DFlashBatchDraft(Array<Model> models, LogitProcessor logit_processor,
                                            Sampler sampler,
                                            std::vector<ModelWorkspace> model_workspaces,
                                            DraftTokenWorkspaceManager draft_token_workspace_manager,
                                            EngineConfig engine_config,
                                            Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(tvm::ffi::make_object<DFlashBatchDraftActionObj>(
      std::move(models), std::move(logit_processor), std::move(sampler),
      std::move(model_workspaces), std::move(draft_token_workspace_manager),
      std::move(engine_config), std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
