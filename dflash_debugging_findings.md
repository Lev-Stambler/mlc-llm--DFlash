# DFlash Speculative Decoding: Debugging Findings

## Problem

DFlash speculative decoding on EC2 g6.2xlarge (L4 GPU) produced **wrong output** with **0% draft token acceptance**:
- All draft tokens were `0,0,0,0,...,0` (token ID 0 = padding)
- DFlash was 2.4x **slower** than baseline (2.93s vs 1.27s for 64 tokens)
- `LOG(FATAL)` debug traps crashed the server after the first verify/decode step

## Root Cause: Float16 Overflow in `project_target_hidden`

The DFlash draft model's `project_target_hidden` function performs a matrix multiplication
that reduces concatenated target hidden states from 20480 dimensions (5 layers x 4096) to
4096 dimensions. This matmul overflows in float16.

### The Overflow Chain

1. **Target hidden states are valid**: max_abs ~57.75, no NaN, no zeros
2. **FC matmul accumulates 20480 products**: with values up to ~57.75, partial sums can exceed float16 max (65504)
3. **Overflow produces Inf**: float16 cannot represent values > 65504
4. **RMSNorm on Inf produces NaN**: `Inf / sqrt(mean(Inf^2)) = Inf / Inf = NaN`
5. **NaN propagates through draft model**: all 4096 hidden dims become either NaN (7) or 0 (4089)
6. **Logits become NaN**: argmax of NaN = token 0 (padding token)
7. **All draft tokens = 0**: verification rejects all, acceptance rate = 0%

### Diagnostic Evidence

| Checkpoint | Values | Status |
|-----------|--------|--------|
| target_hidden (from target model prefill) | nan=0, zero=0, max_abs=57.75 | VALID |
| projected_target_hidden (after FC+RMSNorm) | nan=7/4096, zero=4089/4096 | CORRUPTED |
| draft_forward with controlled 0.1 inputs | nan=0 | VALID (model/weights fine) |
| draft_forward with actual corrupted inputs | nan=256/256 | NaN propagated |

## Fix

**File**: `python/mlc_llm/model/dflash/dflash_model.py`, method `project_target_hidden`

**Before** (original code):
```python
def project_target_hidden(self, target_hidden):
    return self.hidden_norm(self.fc(target_hidden))
```

**After** (float32 matmul + float32 RMSNorm):
```python
def project_target_hidden(self, target_hidden):
    w = self.fc.weight.astype("float32")
    h = target_hidden.astype("float32")
    projected = op.matmul(h, op.permute_dims(w, [1, 0]))
    # RMSNorm in float32: matmul output can exceed fp16 max (65504)
    gamma = self.hidden_norm.weight.astype("float32")
    normalized = op.rms_norm(projected, weight=gamma, axes=[-1], epsilon=self.hidden_norm.epsilon)
    return normalized.astype(self.dtype)
```

**Key insight**: Two-stage overflow. First, the dot product of 20480 large values overflows
float16 during accumulation. Second, even if we do the matmul in float32, the *result* can
still exceed 65504, so casting back to float16 before RMSNorm gives Inf again. The fix keeps
everything in float32 through RMSNorm, which normalizes values to a safe range before the
final cast to float16.

## Results After Fix

- Draft tokens are now real tokens (e.g., `1056,382,14374,...` instead of `0,0,0,...`)
- Draft token acceptance rate: variable, up to 40% per round (6/15 accepted in best round)
- Output matches baseline (deterministic parity confirmed at temperature=0)
- Memory overhead: ~16 MB more (528 MB vs 512 MB for `project_target_hidden` temp buffer)

### Performance

| Mode | Time (64 tokens) | Tokens/sec |
|------|------------------|------------|
| Baseline | 1.29s | ~50 tok/s |
| DFlash | 3.09s | ~21 tok/s |

DFlash is currently ~2.4x slower than baseline. The draft model produces correct tokens but
acceptance rate is not yet high enough to offset the overhead of running the draft forward
pass + verification. Improving draft quality (better model weights, training, or architecture
alignment) would be needed to achieve speedup.

## Other Bugs Fixed (Prior Commits)

- **mstate index** (885a6d30): Draft wrote to mstates[0], verify read from mstates[1]
- **Fallback hidden states**: Zero-initialized dummy tensors when target model lacks hidden state extraction
- **Loader prefix**: Extra `model.` prefix in dflash weight loader
- **logit_pos_arr_ too small**: Buffer was [1]-sized in interactive mode but scatter/gather needs [16+]

## Files Changed

| File | Change |
|------|--------|
| `python/mlc_llm/model/dflash/dflash_model.py` | Float32 matmul + RMSNorm in `project_target_hidden` |
| `cpp/serve/engine_actions/batch_decode.cc` | Removed debug logging |
| `cpp/serve/engine_actions/dflash_batch_draft.cc` | Removed diagnostic blocks |
| `cpp/serve/engine_actions/dflash_batch_verify.cc` | Removed debug logging |
| `cpp/serve/engine_actions/dflash_new_request_prefill.cc` | Removed diagnostic blocks |
| `cpp/serve/model.cc` | Removed diagnostic blocks |
