/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/dflash_commons.h
 * \brief Shared helpers for DFlash engine actions.
 */
#ifndef MLC_LLM_SERVE_ENGINE_ACTIONS_DFLASH_COMMONS_H_
#define MLC_LLM_SERVE_ENGINE_ACTIONS_DFLASH_COMMONS_H_

#include <algorithm>

#include "../engine_state.h"
#include "../model.h"

namespace mlc {
namespace llm {
namespace serve {

inline bool IsDeterministicGenerationConfig(const GenerationConfig& generation_cfg) {
  return generation_cfg->temperature < 1e-5;
}

inline Tensor CopyTensorToCPU(const Tensor& src) {
  if (src->device.device_type == kDLCPU) {
    return src;
  }
  return src.CopyTo(DLDevice{kDLCPU, 0});
}

inline int32_t ArgmaxTokenId(const Tensor& cpu_tensor, int64_t row_index) {
  TVM_FFI_ICHECK_EQ(cpu_tensor->device.device_type, kDLCPU);
  TVM_FFI_ICHECK_EQ(cpu_tensor->ndim, 2);
  TVM_FFI_ICHECK(cpu_tensor.IsContiguous());
  TVM_FFI_ICHECK_EQ(cpu_tensor.DataType(), DataType::Float(32));
  TVM_FFI_ICHECK_GE(row_index, 0);
  TVM_FFI_ICHECK_LT(row_index, cpu_tensor->shape[0]);

  int64_t vocab_size = cpu_tensor->shape[1];
  const float* row = static_cast<const float*>(cpu_tensor->data) + row_index * vocab_size;
  int32_t argmax_token_id = 0;
  float argmax_value = row[0];
  for (int64_t token_id = 1; token_id < vocab_size; ++token_id) {
    if (row[token_id] > argmax_value) {
      argmax_value = row[token_id];
      argmax_token_id = static_cast<int32_t>(token_id);
    }
  }
  return argmax_token_id;
}

inline int64_t GetDataTypeBytes(DLDataType dtype) {
  return (dtype.bits * dtype.lanes + 7) / 8;
}

inline void CopyTensorRows(const Tensor& src, int64_t src_offset, int64_t num_rows,
                           const Tensor& dst, int64_t dst_offset) {
  if (num_rows == 0) {
    return;
  }
  TVM_FFI_ICHECK_EQ(src->ndim, 3);
  TVM_FFI_ICHECK_EQ(dst->ndim, 3);
  TVM_FFI_ICHECK_EQ(src->shape[0], 1);
  TVM_FFI_ICHECK_EQ(dst->shape[0], 1);
  TVM_FFI_ICHECK_EQ(src->shape[2], dst->shape[2]);
  TVM_FFI_ICHECK_EQ(src->dtype, dst->dtype);

  int64_t hidden_size = src->shape[2];
  int64_t bytes_per_row = hidden_size * GetDataTypeBytes(src->dtype);

  DLTensor src_dl = *src.operator->();
  DLTensor dst_dl = *dst.operator->();
  int64_t copy_shape[] = {1, num_rows, hidden_size};
  src_dl.shape = copy_shape;
  dst_dl.shape = copy_shape;
  src_dl.byte_offset += src_offset * bytes_per_row;
  dst_dl.byte_offset += dst_offset * bytes_per_row;
  Tensor::CopyFromTo(&src_dl, &dst_dl, nullptr);
}

inline void ResetProjectedHiddenState(DFlashProjectedHiddenState* state, const Tensor& src,
                                      int64_t src_offset, int64_t num_rows) {
  if (num_rows == 0) {
    state->storage = Tensor{nullptr};
    state->length = 0;
    state->capacity = 0;
    return;
  }
  TVM_FFI_ICHECK_EQ(src->ndim, 3);
  Tensor storage = Tensor::Empty({1, num_rows, src->shape[2]}, src->dtype, src->device);
  CopyTensorRows(src, src_offset, num_rows, storage, /*dst_offset=*/0);
  state->storage = storage;
  state->length = num_rows;
  state->capacity = num_rows;
}

inline void ReserveProjectedHiddenState(DFlashProjectedHiddenState* state, int64_t required_length) {
  TVM_FFI_ICHECK(state->defined());
  if (state->capacity >= required_length) {
    return;
  }

  int64_t new_capacity = std::max(required_length, std::max<int64_t>(state->capacity * 2, 1));
  Tensor new_storage = Tensor::Empty({1, new_capacity, state->storage->shape[2]},
                                     state->storage->dtype, state->storage->device);
  CopyTensorRows(state->storage, /*src_offset=*/0, state->length, new_storage, /*dst_offset=*/0);
  state->storage = new_storage;
  state->capacity = new_capacity;
}

inline void AppendProjectedHiddenRows(DFlashProjectedHiddenState* state, const Tensor& src,
                                      int64_t src_offset, int64_t num_rows) {
  if (num_rows == 0) {
    return;
  }
  if (!state->defined()) {
    ResetProjectedHiddenState(state, src, src_offset, num_rows);
    return;
  }

  ReserveProjectedHiddenState(state, state->length + num_rows);
  CopyTensorRows(src, src_offset, num_rows, state->storage, state->length);
  state->length += num_rows;
}

inline DFlashProjectedHiddenState CloneProjectedHiddenState(
    const DFlashProjectedHiddenState& source_state) {
  DFlashProjectedHiddenState cloned_state;
  if (!source_state.defined()) {
    return cloned_state;
  }
  ResetProjectedHiddenState(&cloned_state, source_state.View(), /*src_offset=*/0,
                            source_state.length);
  return cloned_state;
}

inline Tensor ExtractProjectedFrontierHiddenStates(EngineState estate, const Model& target_model,
                                                   const Model& draft_model,
                                                   const std::vector<int64_t>& base_seq_ids,
                                                   const std::vector<int32_t>& frontier_token_ids) {
  TVM_FFI_ICHECK_EQ(base_seq_ids.size(), frontier_token_ids.size());
  if (base_seq_ids.empty()) {
    return Tensor{nullptr};
  }

  std::vector<int64_t> temp_seq_ids;
  temp_seq_ids.reserve(base_seq_ids.size());
  for (int64_t base_seq_id : base_seq_ids) {
    int64_t temp_seq_id = estate->id_manager.GetNewId();
    target_model->ForkSequence(base_seq_id, temp_seq_id);
    temp_seq_ids.push_back(temp_seq_id);
  }

  ObjectRef frontier_embeddings =
      target_model->TokenEmbed({IntTuple{frontier_token_ids.begin(), frontier_token_ids.end()}});
  auto [frontier_logits, frontier_target_hidden] =
      target_model->BatchDecodeWithHiddenStates(frontier_embeddings, temp_seq_ids);
  static_cast<void>(frontier_logits);
  ObjectRef frontier_project_input = frontier_target_hidden;
  if (!frontier_target_hidden->IsInstance<DRefObj>()) {
    Tensor frontier_hidden_nd = Downcast<Tensor>(frontier_target_hidden);
    TVM_FFI_ICHECK_EQ(frontier_hidden_nd->ndim, 3);
    TVM_FFI_ICHECK_EQ(frontier_hidden_nd->shape[1], 1);
    frontier_project_input = frontier_hidden_nd.CreateView(
        {1, frontier_hidden_nd->shape[0], frontier_hidden_nd->shape[2]}, frontier_hidden_nd->dtype);
  }
  Tensor frontier_projected =
      Downcast<Tensor>(draft_model->ProjectTargetHiddenStates(frontier_project_input));

  for (int64_t temp_seq_id : temp_seq_ids) {
    target_model->RemoveSequence(temp_seq_id);
    estate->id_manager.RecycleId(temp_seq_id);
  }
  return frontier_projected;
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_ENGINE_ACTIONS_DFLASH_COMMONS_H_
