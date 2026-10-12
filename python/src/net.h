#pragma once
#include <onnxruntime_cxx_api.h>
#include "board.h"

namespace ataxx {

struct Net {
  Ort::Env env;
  Ort::Session session{nullptr};
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  explicit Net(const char* path) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    options.AddConfigEntry("session.set_denormal_as_zero", "1");
    session = Ort::Session(env, path, options);
  }

  void eval(float* feat, float* policy, float* value) {
    int64_t shape[4] = {1, PLANES, 7, 7};
    Ort::Value input = Ort::Value::CreateTensor<float>(memory, feat, PLANES * SQ, shape, 4);
    const char* in[] = {"x"};
    const char* out[] = {"policy", "value"};
    auto result = session.Run(Ort::RunOptions{nullptr}, in, &input, 1, out, 2);
    std::memcpy(policy, result[0].GetTensorData<float>(), NMOVES * sizeof(float));
    std::memcpy(value, result[1].GetTensorData<float>(), 4 * sizeof(float));
  }
};

}
