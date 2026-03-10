/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/dflash_new_request_prefill.cc
 * \brief DFlash prefill action: prefill the target model, extract intermediate
 *  hidden states, project them, and run one-step block draft.
 */

#include "../sampler/sampler.h"
#include "batch_prefill_base.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The action that prefills requests for DFlash speculative decoding.
 *  Model 0 = target model, Model 1 = DFlash draft model (stateless, no KV cache).
 */
class DFlashNewRequestPrefillActionObj : public BatchPrefillBaseActionObj {
 public:
  explicit DFlashNewRequestPrefillActionObj(
      Array<Model> models, LogitProcessor logit_processor, Sampler sampler,
      std::vector<ModelWorkspace> model_workspaces,
      DraftTokenWorkspaceManager draft_token_workspace_manager, EngineConfig engine_config,
      std::vector<tvm::ffi::json::Object> model_configs,
      Optional<EventTraceRecorder> trace_recorder)
      : BatchPrefillBaseActionObj(std::move(models), std::move(engine_config),
                                  std::move(model_configs), std::move(trace_recorder)),
        logit_processor_(std::move(logit_processor)),
        sampler_(std::move(sampler)),
        model_workspaces_(std::move(model_workspaces)),
        draft_token_workspace_manager_(std::move(draft_token_workspace_manager)) {}

  Array<Request> Step(EngineState estate) final {
    // - Find the requests in `waiting_queue` that can prefill in this step.
    std::vector<PrefillInput> prefill_inputs;
    {
      NVTXScopedRange nvtx_scope("DFlashPrefill getting requests");
      prefill_inputs = GetRequestStateEntriesToPrefill(estate);
      if (prefill_inputs.empty()) {
        return {};
      }
    }

    int num_rsentries = prefill_inputs.size();
    {
      NVTXScopedRange nvtx_scope("DFlashPrefill matching prefix");
      for (int i = 0; i < num_rsentries; ++i) {
        MatchPrefixCache(estate, &prefill_inputs[i]);
      }
    }

    auto tstart = std::chrono::high_resolution_clock::now();

    // - Update status of request states from pending to alive.
    Array<String> request_ids;
    std::vector<RequestState> rstates_of_entries;
    std::vector<RequestStateStatus> status_before_prefill;
    UpdateRequestToAlive(prefill_inputs, estate, &request_ids, &rstates_of_entries,
                         &status_before_prefill);

    // - Get embedding and run prefill for target model (model 0) only.
    //   DFlash draft model (model 1) has no KV cache and no separate prefill.
    std::vector<int> prefill_lengths;
    prefill_lengths.resize(/*size=*/num_rsentries, /*value=*/-1);
    Tensor logits_for_sample{nullptr};
    std::unordered_map<int, std::unordered_set<int>> fork_rsentry_child_map;

    // Model 0 (target): compute embeddings, prefill, get hidden states
    {
      int model_id = 0;
      std::vector<int64_t> request_internal_ids;
      request_internal_ids.reserve(num_rsentries);
      ObjectRef embeddings = model_workspaces_[model_id].embeddings;
      int cum_prefill_length = 0;
      bool single_input =
          num_rsentries == 1 && prefill_inputs[0].rsentry->mstates[model_id]->inputs.size() == 1;

      for (int i = 0; i < num_rsentries; ++i) {
        const RequestStateEntry& rsentry = prefill_inputs[i].rsentry;
        RequestModelState mstate = rsentry->mstates[model_id];
        TVM_FFI_ICHECK(mstate->draft_output_tokens.empty());
        TVM_FFI_ICHECK(mstate->draft_token_slots.empty());
        if (status_before_prefill[i] == RequestStateStatus::kPending) {
          if (!estate->prefix_cache->HasSequence(mstate->internal_id)) {
            if (rsentry->parent_idx == -1) {
              models_[model_id]->AddNewSequence(mstate->internal_id);
            } else {
              models_[model_id]->ForkSequence(rstates_of_entries[i]
                                                  ->entries[rsentry->parent_idx]
                                                  ->mstates[model_id]
                                                  ->internal_id,
                                              mstate->internal_id);
            }
          }
          if (rsentry->child_indices.empty()) {
            models_[model_id]->EnableSlidingWindowForSeq(mstate->internal_id);
          }
        }
        request_internal_ids.push_back(mstate->internal_id);

        auto [input_data, input_length] =
            ChunkPrefillInputData(mstate, prefill_inputs[i].max_prefill_length);
        if (prefill_lengths[i] == -1) {
          prefill_lengths[i] = input_length;
        }
        mstate->num_prefilled_tokens += input_length;
        mstate->prefilled_inputs.push_back(input_data[0]);

        RECORD_EVENT(trace_recorder_, rsentry->request->id, "start embedding");
        for (int j = 0; j < static_cast<int>(input_data.size()); ++j) {
          embeddings = input_data[j]->GetEmbedding(
              models_[model_id],
              /*dst=*/!single_input ? &model_workspaces_[model_id].embeddings : nullptr,
              /*offset=*/cum_prefill_length);
          cum_prefill_length += input_data[j]->GetLength();
        }
        RECORD_EVENT(trace_recorder_, rsentry->request->id, "finish embedding");
      }

      RECORD_EVENT(trace_recorder_, request_ids, "start prefill");

      // Run target model prefill with intermediate hidden state extraction.
      // BatchPrefillWithHiddenStates returns (logits_at_logit_positions, target_hidden).
      auto [prefill_logits, target_hidden] = models_[model_id]->BatchPrefillWithHiddenStates(
          embeddings, request_internal_ids, prefill_lengths);
      RECORD_EVENT(trace_recorder_, request_ids, "finish prefill");

      // Commit prefix cache changes
      estate->prefix_cache->CommitSequenceExtention();

      // Reshape logits from [1, num_sequences, v] to [num_sequences, v] for sampling.
      logits_for_sample = prefill_logits.CreateView(
          {prefill_logits->shape[1], prefill_logits->shape[2]}, prefill_logits->dtype);

      // Project target hidden states via draft model and store per-request.
      // target_hidden: [1, total_seq_len, num_layers * hidden_size]
      // projected:     [1, total_seq_len, hidden_size]
      RECORD_EVENT(trace_recorder_, request_ids, "start project target hidden");
      ObjectRef projected = models_[1]->ProjectTargetHiddenStates(target_hidden);
      RECORD_EVENT(trace_recorder_, request_ids, "finish project target hidden");

      // Store projected target hidden per request.
      // TODO(dflash): For multi-request prefill, slice the projected tensor per request.
      // For now, single-request is the common case and storing the full tensor works.
      for (int i = 0; i < num_rsentries; ++i) {
        model_workspaces_[0].dflash_projected_target_hidden[request_internal_ids[i]] = projected;
      }
    }

    // - Prepare sampling configurations and sample.
    Array<String> child_request_ids;
    std::vector<int> child_sample_indices;
    std::vector<RequestStateEntry> rsentries_for_sample;
    std::vector<RandomGenerator*> rngs;
    std::vector<bool> rsentry_activated;
    Array<GenerationConfig> child_generation_cfg;
    child_sample_indices.reserve(num_rsentries);
    child_generation_cfg.reserve(num_rsentries);
    child_request_ids.reserve(num_rsentries);
    rsentries_for_sample.reserve(num_rsentries);
    rngs.reserve(num_rsentries);
    rsentry_activated.reserve(num_rsentries);
    for (int i = 0; i < num_rsentries; ++i) {
      const RequestStateEntry& rsentry = prefill_inputs[i].rsentry;
      if (!rsentry->mstates[0]->inputs.empty()) {
        continue;
      }
      int remaining_num_child_to_activate = prefill_inputs[i].num_child_to_activate;
      for (int child_idx : rsentry->child_indices) {
        if ((rstates_of_entries[i]->entries[child_idx]->status == RequestStateStatus::kPending &&
                 rstates_of_entries[i]
                     ->entries[child_idx]
                     ->mstates[0]
                     ->committed_tokens.empty() ||
             fork_rsentry_child_map[i].count(child_idx))) {
          fork_rsentry_child_map[i].insert(child_idx);
          child_sample_indices.push_back(i);
          rsentries_for_sample.push_back(rstates_of_entries[i]->entries[child_idx]);
          child_request_ids.push_back(rsentry->request->id);
          child_generation_cfg.push_back(rsentry->request->generation_cfg);
          rngs.push_back(&rstates_of_entries[i]->entries[child_idx]->rng);
          if (remaining_num_child_to_activate == 0) {
            rsentry_activated.push_back(false);
            continue;
          }
          rsentry_activated.push_back(true);
          --remaining_num_child_to_activate;
          rstates_of_entries[i]->entries[child_idx]->status = RequestStateStatus::kAlive;
          int64_t child_internal_id =
              rstates_of_entries[i]->entries[child_idx]->mstates[0]->internal_id;
          models_[0]->ForkSequence(rsentry->mstates[0]->internal_id, child_internal_id);
          if (rstates_of_entries[i]->entries[child_idx]->child_indices.empty()) {
            models_[0]->EnableSlidingWindowForSeq(child_internal_id);
          }
        }
      }
      if (rsentry->child_indices.empty()) {
        child_sample_indices.push_back(i);
        rsentries_for_sample.push_back(rsentry);
        child_request_ids.push_back(rsentry->request->id);
        child_generation_cfg.push_back(rsentry->request->generation_cfg);
        rngs.push_back(&rsentry->rng);
        rsentry_activated.push_back(true);
      }
    }

    // Prepare logit processor input
    Array<GenerationConfig> generation_cfg;
    Array<RequestModelState> mstates_for_logitproc;
    std::vector<int> sample_indices(num_rsentries);
    generation_cfg.reserve(num_rsentries);
    mstates_for_logitproc.reserve(num_rsentries);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);
    for (int i = 0; i < num_rsentries; ++i) {
      generation_cfg.push_back(prefill_inputs[i].rsentry->request->generation_cfg);
      mstates_for_logitproc.push_back(prefill_inputs[i].rsentry->mstates[0]);
    }

    const auto& [renormalized_probs, sample_results] = ApplyLogitProcessorAndSample(
        logit_processor_, sampler_, logits_for_sample, generation_cfg, request_ids,
        mstates_for_logitproc, rngs, sample_indices, child_generation_cfg, child_request_ids,
        child_sample_indices);
    UpdateRequestStateEntriesWithSampleResults(rsentries_for_sample, rsentry_activated,
                                               sample_results);

    // Update engine metrics
    auto tend = std::chrono::high_resolution_clock::now();
    estate->metrics.engine_prefill_time_sum += static_cast<double>((tend - tstart).count()) / 1e9;

    Array<Request> processed_requests;
    processed_requests.reserve(num_rsentries);
    for (int i = 0; i < num_rsentries; ++i) {
      processed_requests.push_back(prefill_inputs[i].rsentry->request);
    }

    return processed_requests;
  }

 private:
  /*! \brief The logit processor. */
  LogitProcessor logit_processor_;
  /*! \brief The sampler to sample new tokens. */
  Sampler sampler_;
  /*! \brief The model workspaces. */
  std::vector<ModelWorkspace> model_workspaces_;
  /*! \brief The draft token workspace manager. */
  DraftTokenWorkspaceManager draft_token_workspace_manager_;
};

EngineAction EngineAction::DFlashNewRequestPrefill(
    Array<Model> models, LogitProcessor logit_processor, Sampler sampler,
    std::vector<ModelWorkspace> model_workspaces,
    DraftTokenWorkspaceManager draft_token_workspace_manager, EngineConfig engine_config,
    std::vector<tvm::ffi::json::Object> model_configs,
    Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(make_object<DFlashNewRequestPrefillActionObj>(
      std::move(models), std::move(logit_processor), std::move(sampler),
      std::move(model_workspaces), std::move(draft_token_workspace_manager),
      std::move(engine_config), std::move(model_configs), std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
