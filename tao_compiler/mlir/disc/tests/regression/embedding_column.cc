// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/mlir/disc/tests/mlir_feature_test.h"
#include "tensorflow/compiler/mlir/disc/tests/mlir_test.h"
#include "tensorflow/core/platform/test.h"

namespace mlir_test {

const std::string c_ft_path =
    "tensorflow/compiler/mlir/disc/tests/regression/data/";

TEST(EmbeddingColumnTest, DynamicShape3DI64Test) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "embedding_column_d_3d_i64.mlir",
      /*backend_types*/ {kSupportedCPUBackendList},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x4xi64_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {{0, 6, 0, 0, 8, 4, 4, 8, 0, 7, 6, 9,
                       6, 0, 8, 9, 0, 0, 0, 0, 0, 0, 9, 4}}));
}

}  // namespace mlir_test
