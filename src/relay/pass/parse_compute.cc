/*!
 * Copyright (c) 2019 by Contributors
 * \file parse_compute.cc
 * \brief Parse relay index expression to tvm expression
 */
#include <tvm/ir_operator.h>
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/ie_mixture.h>
#include "../../pass/ir_util.h"

namespace tvm {
namespace relay {

#define TRANSLATE_BINARY_OP(name, make_func)                         \
  op_translate.insert(std::make_pair(Op::Get(name).operator->(),     \
                      [](std::vector<tvm::Expr> args) -> tvm::Expr { \
                        return make_func(args[0], args[1]);          \
                      }));                                           \

#define TRANSLATE_REDUCTION_OP(name, make_func)                      \
  op_translate.insert(std::make_pair(Op::Get(name).operator->(),     \
                      [&](std::vector<tvm::Expr> args) -> tvm::Expr {\
                        return make_func(args[0], *reduce_axis_);    \
                      }));

class Relay2TVMConverter : public ExprFunctor<tvm::Expr(const Expr&)> {
 public:
  Relay2TVMConverter() {
    TRANSLATE_BINARY_OP("add", tvm::ir::Add::make);
    TRANSLATE_BINARY_OP("subtract", tvm::ir::Sub::make);
    TRANSLATE_BINARY_OP("multiply", tvm::ir::Mul::make);
    TRANSLATE_BINARY_OP("divide", tvm::ir::Div::make);
    TRANSLATE_BINARY_OP("mod", tvm::ir::Mod::make);

    TRANSLATE_REDUCTION_OP("SUM", sum);
    TRANSLATE_REDUCTION_OP("MAX", max);
    TRANSLATE_REDUCTION_OP("MIN", max);
  }


  tvm::Expr Convert(const Expr &expr,
                    Map<relay::Var, tvm::Tensor>* tensor_map,
                    const std::unordered_map<relay::Var, tvm::Var, NodeHash, NodeEqual>& itervar_map,
                    const tvm::Array<IterVar>& reduce_axis) {
    tensor_map_ = tensor_map;
    itervar_map_ = &itervar_map;
    reduce_axis_ = &reduce_axis;
    return this->VisitExpr(expr);
  }

  tvm::Expr VisitExpr_(const VarNode *op) final {
    Var var = GetRef<Var>(op);
    const TensorTypeNode* type = op->checked_type().as<TensorTypeNode>();
    if (type->shape.size() == 0) {
      return itervar_map_->at(var);
    } else {
      if (tensor_map_->count(var) == 0) {
        tensor_map_->Set(var,
                     PlaceholderOpNode::make(op->name_hint(), type->shape, type->dtype).output(0));
      }

      Tensor tensor = tensor_map_->at(var);
      return tvm::ir::Call::make(type->dtype, op->name_hint(),
                                 Array<tvm::Expr>{nullptr}, tvm::ir::Call::CallType::Halide,
                                 tensor->op, 0);
    }
  }

  tvm::Expr VisitExpr_(const IndexNode *op) final {
    tvm::Expr base = VisitExpr(op->base);
    Array<tvm::Expr> indices;
    for (auto x : op->indices) {
      indices.push_back(VisitExpr(x));
    }
    const tvm::ir::Call* call = base.as<tvm::ir::Call>();
    CHECK(call != nullptr);
    return tvm::ir::Call::make(call->type, call->name, indices,
                               call->call_type, call->func, call->value_index);
  }

  tvm::Expr VisitExpr_(const ConstantNode *n) final {
    const TensorTypeNode *type = n->checked_type().as<TensorTypeNode>();
    CHECK_EQ(type->shape.size(), 0) << "Only accept scalar expression";

    DLPACK_TYPE_SWITCH(tvm::ir::HalideType2DLTensorType(type->dtype), Dtype, {
       return tvm::make_const<Dtype>(type->dtype, static_cast<Dtype*>(n->data->data)[0]);
    });
    return tvm::Expr();
  }

  tvm::Expr VisitExpr_(const CallNode *n) final {
    const OpNode* op = n->op.as<OpNode>();
    CHECK(op) << "Cannot convert " << GetRef<Call>(n) << " to tvm expression";

    // binary ops
    auto x = op_translate.find(op);
    if (x != op_translate.end()) {
      std::vector<tvm::Expr> args;
      for (auto x : n->args) {
        args.push_back(VisitExpr(x));
      }
      return x->second(args);
    }

    // Todo(lmzheng): add more ops here
    LOG(FATAL) << "Cannot convert " << GetRef<Call>(n) << " to tvm expression";
    return tvm::Expr();
  }

 private:
  std::unordered_map<const OpNode*, std::function<tvm::Expr(std::vector<tvm::Expr>&)> > op_translate;

  Map<relay::Var, tvm::Tensor>* tensor_map_;
  const std::unordered_map<relay::Var, tvm::Var, NodeHash, NodeEqual>* itervar_map_;
  const tvm::Array<IterVar>* reduce_axis_;
};

class ComputeParser : public ExprMutator {
 public:
  Expr VisitExpr_(const LetNode* n) final {
    ty_map_[n->var] = n->value;
    return ExprMutator::VisitExpr_(n);
  }

  Expr VisitExpr_(const CallNode* n) final {
    static const Op& compute = Op::Get("compute");
    Call new_n = Downcast<Call>(ExprMutator::VisitExpr_(n));

    if (!new_n->op.same_as(compute)) {
      return new_n;
    }

    const FunctionNode* func = new_n->args[0].as<FunctionNode>();
    if (func == nullptr) {
      func = PropagateLet_(new_n->args[0]);
    }
    CHECK(func != nullptr) << "Cannot parse " << new_n->args[0]
                           << " as an index expression";

    const ComputeAttrs* attrs = new_n->attrs.as<ComputeAttrs>();

    // make compute op
    std::string name = "compute";
    Array<IterVar> axis;
    Array<IterVar> reduce_axis;
    size_t n_spatial = attrs->shape.size();
    size_t n_reduction = func->params.size() - n_spatial;
    std::unordered_map<relay::Var, tvm::Var, NodeHash, NodeEqual> itervar_map;

    for (size_t i = 0; i < n_spatial; ++i) {
      axis.push_back(Param2var_(func->params[i], IterVarType::kDataPar,
                                attrs->shape[i], itervar_map));
    }
    for (size_t i = 0; i < n_reduction; ++i) {
      reduce_axis.push_back(Param2var_(func->params[n_spatial + i], IterVarType::kCommReduce,
                                       attrs->reduction[i], itervar_map));
    }

    tvm::Map<relay::Var, tvm::Tensor> tensor_map;
    tvm::Expr expr = converter_.Convert(func->body, &tensor_map, itervar_map, reduce_axis);

    Operation operation = ComputeOpNode::make(
        name, "", Map<std::string, NodeRef>(nullptr),
        axis, Array<tvm::Expr>{expr});

    // new attributes
    auto new_attrs = make_node<ParsedTVMOpAttrs>();
    tvm::Array<Expr> args;
    tvm::Array<Tensor> inputs;

    for (auto x : tensor_map) {
      args.push_back(x.first);
      inputs.push_back(x.second);
    }
    new_attrs->inputs = inputs;
    new_attrs->outputs = Array<Tensor>{operation.output(0)};

    return CallNode::make(Op::Get("parsed_tvm_op"),
                          Array<Expr>{TupleNode::make(args)}, Attrs(new_attrs));
  }

 private:
  IterVar Param2var_(const Var& param, IterVarType iter_type, tvm::Expr extent,
                     std::unordered_map<relay::Var, tvm::Var, NodeHash, NodeEqual> &itervar_map) {
    const TensorTypeNode* p_type = param->checked_type().as<TensorTypeNode>();
    CHECK(p_type != nullptr);
    CHECK_EQ(p_type->shape.size(), 0) << "The index should be scalar";

    tvm::Var var(param->name_hint(), p_type->dtype);
    IterVar iter_var = IterVarNode::make(
        Range::make_by_min_extent(0, extent),
        var, iter_type);

    itervar_map[param] = var;
    return iter_var;
  }

  const FunctionNode* PropagateLet_(const Expr &var) const {
    // handle let chain
    auto x = ty_map_.find(var);
    while (x != ty_map_.end()) {
      const FunctionNode* ret = x->second.as<FunctionNode>();
      if (ret != nullptr) { return ret; }
      x = ty_map_.find(x->second);
    }
    return nullptr;
  }

  std::unordered_map<Expr, Expr, NodeHash, NodeEqual> ty_map_;
  Relay2TVMConverter converter_;
};

Expr ParseCompute(const Expr& e) {
  return ComputeParser().Mutate(e);
}

TVM_REGISTER_API("relay._ir_pass.parse_compute")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = ParseCompute(args[0]);
});

}  // namespace relay
}  // namespace tvm
