// RUN: disc-opt -disc-dynamic-slice-converter -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: func.func @dynamic_slice
func.func @dynamic_slice(%arg0: tensor<?xi64>, %arg1: tensor<i32>) -> tensor<1xi64> {
  %out = "mhlo.dynamic_slice"(%arg0, %arg1) {
    slice_sizes = dense<1> : tensor<1xi64>
  } : (tensor<?xi64>, tensor<i32>) -> tensor<1xi64>
  return %out : tensor<1xi64>
}
