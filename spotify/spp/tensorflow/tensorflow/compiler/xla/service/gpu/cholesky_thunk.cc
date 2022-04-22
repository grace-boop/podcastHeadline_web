/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/gpu/cholesky_thunk.h"

#include <complex>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/gpu/precompiled_kernels.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/device_memory.h"

namespace xla {
namespace gpu {

namespace {

StatusOr<GpuSolverContext*> GetContext(se::Stream* stream) {
  // TODO(b/214454412): This global hashtable is incorrect (ABA bug if a Stream
  // is added to the hasthable, then deleted, and then a new Stream is created
  // at the same address).  It also leaks memory!
  static absl::Mutex mu(absl::kConstInit);
  static auto contexts =
      new absl::flat_hash_map<se::Stream*, GpuSolverContext> ABSL_GUARDED_BY(
          mu);

  absl::MutexLock lock(&mu);
  auto result = contexts->emplace(stream, GpuSolverContext());
  if (result.second) {
    TF_ASSIGN_OR_RETURN(result.first->second, GpuSolverContext::Create(stream));
  }
  return &result.first->second;
}

}  // namespace

CholeskyThunk::CholeskyThunk(ThunkInfo thunk_info,
                             const CholeskyOptions& options,
                             const se::GpuAsmOpts asm_opts,
                             BufferAllocation::Slice a_buffer,
                             BufferAllocation::Slice workspace_buffer,
                             BufferAllocation::Slice info_buffer,
                             PrimitiveType type, int64_t batch_size, int64_t n)
    : Thunk(Kind::kCholesky, thunk_info),
      asm_opts_(asm_opts),
      uplo_(options.lower() ? se::blas::UpperLower::kLower
                            : se::blas::UpperLower::kUpper),
      a_buffer_(a_buffer),
      workspace_buffer_(workspace_buffer),
      info_buffer_(info_buffer),
      type_(type),
      batch_size_(batch_size),
      n_(n) {}

Status CholeskyThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(3) << "type=" << PrimitiveType_Name(type_)
          << " uplo=" << se::blas::UpperLowerString(uplo_)
          << " batch_size=" << batch_size_ << " n=" << n_
          << " a=" << a_buffer_.ToString()
          << " workspace=" << workspace_buffer_.ToString()
          << " info=" << info_buffer_.ToString();

  TF_ASSIGN_OR_RETURN(GpuSolverContext * context, GetContext(params.stream));
  if (context->SupportsPotrfBatched()) {
    switch (type_) {
      case F32:
        return DoPotrfBatched<float>(params, context);
      case F64:
        return DoPotrfBatched<double>(params, context);
      case C64:
        return DoPotrfBatched<std::complex<float>>(params, context);
      case C128:
        return DoPotrfBatched<std::complex<double>>(params, context);
      default:
        return InvalidArgument("Invalid type for cholesky %s",
                               PrimitiveType_Name(type_));
    }
  } else {
    switch (type_) {
      case F32:
        return DoPotrfUnbatched<float>(params, context);
      case F64:
        return DoPotrfUnbatched<double>(params, context);
      case C64:
        return DoPotrfUnbatched<std::complex<float>>(params, context);
      case C128:
        return DoPotrfUnbatched<std::complex<double>>(params, context);
      default:
        return InvalidArgument("Invalid type for cholesky %s",
                               PrimitiveType_Name(type_));
    }
  }
}

template <typename T>
Status CholeskyThunk::DoPotrfBatched(const ExecuteParams& params,
                                     GpuSolverContext* context) {
  se::Stream* stream = params.stream;
  T* a_base = static_cast<T*>(
      params.buffer_allocations->GetDeviceAddress(a_buffer_).opaque());
  se::DeviceMemory<int> infos(
      params.buffer_allocations->GetDeviceAddress(info_buffer_));
#if TENSORFLOW_USE_ROCSOLVER
  // hipsolver is not supported so allocate a GPU buffer
  se::ScopedDeviceMemory<T*> ptrs =
      stream->parent()->AllocateOwnedArray<T*>(batch_size_);
  auto as = *ptrs;
#else
  se::DeviceMemory<T*> as(
      params.buffer_allocations->GetDeviceAddress(workspace_buffer_));
#endif

  CHECK_GE(as.size(), batch_size_);
  CHECK_GE(infos.size(), batch_size_);

  // Run a kernel that sets as[i] = &a_base[i * stride].
  const int64_t stride_bytes = n_ * n_ * sizeof(T);
  TF_RETURN_IF_ERROR(MakeBatchPointers(
      stream, asm_opts_, se::DeviceMemoryBase(a_base), stride_bytes,
      static_cast<int>(batch_size_), se::DeviceMemoryBase(as)));

  // Now that we've set up the `as` array, we can call cusolver.
  return context->PotrfBatched(uplo_, n_, as, n_, infos, batch_size_);
}

template <typename T>
Status CholeskyThunk::DoPotrfUnbatched(const ExecuteParams& params,
                                       GpuSolverContext* context) {
  T* a_base = static_cast<T*>(
      params.buffer_allocations->GetDeviceAddress(a_buffer_).opaque());
  int* info_base = static_cast<int*>(
      params.buffer_allocations->GetDeviceAddress(info_buffer_).opaque());
  se::DeviceMemoryBase workspace =
      params.buffer_allocations->GetDeviceAddress(workspace_buffer_);

  int64_t stride = n_ * n_;
  for (int64_t i = 0; i < batch_size_; ++i) {
    se::DeviceMemory<T> a_data(
        se::DeviceMemoryBase(&a_base[i * stride], sizeof(T) * stride));
    se::DeviceMemory<int> info_data(
        se::DeviceMemoryBase(&info_base[i], sizeof(int)));
    TF_RETURN_IF_ERROR(
        context->Potrf(uplo_, n_, a_data, n_, info_data, workspace));
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla