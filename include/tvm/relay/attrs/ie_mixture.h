/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/attrs/ie_mixture.h
 * \brief The attribute for compute primitives in relay.
 */
#ifndef TVM_RELAY_ATTRS_TVM_MIXTURE_H_
#define TVM_RELAY_ATTRS_TVM_MIXTURE_H_

#include <tvm/operation.h>
#include <tvm/attrs.h>
#include <tvm/node/container.h>
#include <string>

namespace tvm {
namespace relay {

struct ComputeAttrs : public tvm::AttrsNode<ComputeAttrs> {
  Array<IndexExpr> shape;
  Array<IndexExpr> reduction;

  TVM_DECLARE_ATTRS(ComputeAttrs, "relay.attrs.ComputeAttrs") {
    TVM_ATTR_FIELD(shape)
        .describe("The shape of the tensor");
    TVM_ATTR_FIELD(reduction)
        .describe("The lengths of reduction axis");
  }
};

struct ParsedTVMOpAttrs : public tvm::AttrsNode<ParsedTVMOpAttrs> {
  Array<Tensor> inputs;
  Array<Tensor> outputs;

  TVM_DECLARE_ATTRS(ParsedTVMOpAttrs, "relay.attrs.ParsedTVMOpAttrs") {
    TVM_ATTR_FIELD(inputs)
        .describe("The input tensors");
    TVM_ATTR_FIELD(outputs)
        .describe("The output tensors");
  }
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_ATTRS_TVM_MIXTURE_H_
