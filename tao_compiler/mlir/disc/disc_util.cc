/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/disc/disc_util.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "tensorflow/compiler/mlir/disc/transforms/placement_utils.h"

namespace mlir {
namespace disc_ral {

bool IsSmallBuffer(Value alloc) {
  constexpr unsigned kMaximumSizeInBytes = 128;
  constexpr unsigned kBitwidthOfIndexType = 64;

  auto type = alloc.getType().dyn_cast<ShapedType>();
  if (!type || !type.hasStaticShape()) return false;

  // For index types, use the provided size, as the type does not know.
  unsigned int bitwidth = type.getElementType().isIndex()
                              ? kBitwidthOfIndexType
                              : type.getElementTypeBitWidth();
  return type.getNumElements() * bitwidth <= kMaximumSizeInBytes * 8;
}

bool IsSmallCpuBuffer(Value alloc) {
  if (placement_utils::isGpuMemRef(alloc)) return false;
  return IsSmallBuffer(alloc);
}

bool IsSmallCpuAlloc(Value alloc) {
  return IsSmallCpuBuffer(alloc) && alloc.getDefiningOp<memref::AllocOp>();
}

bool IsOpWriteValue(Operation* op, Value value) {
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 2> effects;
  MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
  // Suppose that value without `MemoryEffectOpInterface` is readonly.
  if (!interface) return false;

  interface.getEffectsOnValue(value, effects);
  return llvm::any_of(
      effects, [](const mlir::MemoryEffects::EffectInstance& instance) {
        return mlir::isa<mlir::MemoryEffects::Write>(instance.getEffect());
      });
}

bool IsMemRefAliasOp(Operation* op) {
  return dyn_cast<ViewLikeOpInterface>(op) != nullptr;
}

Value getRootMemRef(Value memref) {
  Value rootMemRef = memref;
  while (Operation* operandOp = rootMemRef.getDefiningOp()) {
    if (!isa<memref::SubViewOp, memref::ViewOp, memref::CastOp,
             memref::ReinterpretCastOp>(operandOp))
      break;
    rootMemRef = operandOp->getOperand(0);
  }
  return rootMemRef;
}

bool isSameUnderlineBuffer(Value lhs, Value rhs) {
  return getRootMemRef(lhs) == getRootMemRef(rhs);
}

bool parseEinsumEquation(
    llvm::StringRef equation,
    llvm::SmallDenseMap<char, llvm::SmallDenseMap<EquationVariable, size_t>>&
        tokens,
    SmallVector<char>* lhs_original_tokens,
    SmallVector<char>* rhs_original_tokens,
    SmallVector<char>* result_original_tokens) {
  size_t index = 0;
  size_t sub_index = 0;
  EquationVariable current_variable = kIsLhs;
  SmallVector<char> lhs_original_tokens_internal;
  SmallVector<char> rhs_original_tokens_internal;
  SmallVector<char> result_original_tokens_internal;
  bool explicit_result = false;
  while (index < equation.size()) {
    if (std::isalpha(equation[index])) {
      if (current_variable == kIsLhs) {
        tokens[equation[index]][kIsLhs] = sub_index;
        lhs_original_tokens_internal.push_back(equation[index]);
        sub_index++;
      } else if (current_variable == kIsRhs) {
        tokens[equation[index]][kIsRhs] = sub_index;
        rhs_original_tokens_internal.push_back(equation[index]);
        sub_index++;
      } else {
        tokens[equation[index]][kIsResult] = sub_index;
        result_original_tokens_internal.push_back(equation[index]);
        sub_index++;
      }
    } else if (equation.substr(index, 1).contains(",")) {
      current_variable = kIsRhs;
      sub_index = 0;
    } else if ((index < (equation.size() - 1)) &&
               (equation.substr(index, 2).contains("->"))) {
      current_variable = kIsResult;
      explicit_result = true;
      sub_index = 0;
      index++;
    } else if (equation[index] == ' ') {
      // do nothing but continue
    } else {
      return false;
    }
    index++;
  }

  // If no "->" in the equation, deduce the result tokens
  // TODO: handle when operands contain ellipsis
  if (!explicit_result) {
    sub_index = 0;
    for (char lhs_c : lhs_original_tokens_internal) {
      if (std::find(rhs_original_tokens_internal.begin(),
                    rhs_original_tokens_internal.end(),
                    lhs_c) == rhs_original_tokens_internal.end()) {
        tokens[lhs_c][kIsResult] = sub_index;
        result_original_tokens_internal.push_back(lhs_c);
        sub_index++;
      }
    }
    for (char rhs_c : rhs_original_tokens_internal) {
      if (std::find(lhs_original_tokens_internal.begin(),
                    lhs_original_tokens_internal.end(),
                    rhs_c) == lhs_original_tokens_internal.end()) {
        tokens[rhs_c][kIsResult] = sub_index;
        result_original_tokens_internal.push_back(rhs_c);
        sub_index++;
      }
    }
  }
  if (lhs_original_tokens) {
    lhs_original_tokens->swap(lhs_original_tokens_internal);
  }
  if (rhs_original_tokens) {
    rhs_original_tokens->swap(rhs_original_tokens_internal);
  }
  if (result_original_tokens) {
    result_original_tokens->swap(result_original_tokens_internal);
  }
  return true;
}

}  // namespace disc_ral
}  // namespace mlir