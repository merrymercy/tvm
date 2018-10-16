/*!
 *  Copyright (c) 2018 by Contributors
 * \file ast_extractor.cc
 * \brief Extract simplified AST from lowered IR
 */

#include <tvm/api_registry.h>
#include <stack>
#include "feature_visitor.h"
#include "touch_extractor.h"

namespace tvm {
namespace autotvm {

/*!
* \brief Return whether the string `value` ends with string `ending`
* \param value Base string
* \param ending Ending string
*/
inline bool EndsWith(std::string const & value, std::string const & ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

// a node in AST
class Tree {
 public:
  explicit Tree(VarExpr var) {
    name = var.get()->name_hint;
  }

  explicit Tree(std::string node_name) : name(node_name) {
  }

  std::string name;
  std::vector<std::shared_ptr<Tree>> children;
  std::vector<float> additional;
};

// collect all index vars from a buffer index
class IndexvarCollector: public IRVisitor {
 public:
  void Collect(Expr expr) {
    this->Visit(expr);
  }

  void Visit_(const Variable *op) {
    vars.insert(op);
  }

  std::set<const Variable*> vars;
};

// extract simplified ast
class ASTExtractor : public FeatureVisitor {
 public:
  void Extract(Stmt stmt, std::shared_ptr<Tree> root,
               const std::unordered_map<VarExpr, ItervarFeature,
                                        tvm::ExprHash, tvm::ExprEqual> *itervar_map,
               const std::set<TouchedBuffer> *innermost_buffers) {
    root_stack_.push_back(root);
    itervar_map_ = itervar_map;
    innermost_buffers_ = innermost_buffers;
    CHECK_EQ(itervar_map == nullptr, innermost_buffers == nullptr);
    this->Visit(stmt);
  }

 private:
  bool EnterItervar_(VarExpr var, int64_t length, AnnotationType ann_type) {
    if (EndsWith(var.get()->name_hint, ".init")) {
      LOG(FATAL) << "Should never happen!!";
      return false;
    }
    std::shared_ptr<Tree> node = std::make_shared<Tree>("for");

    if (itervar_map_ == nullptr) {  // do not attach statistic feature on tree node
      // length
      node->additional.push_back(static_cast<float>(length));
      // one hot annotation
      for (int i = 0; i < kNum; i++) {
        node->additional.push_back(static_cast<float>
                                   (i == ann_type));
      }
    } else {
      const ItervarFeature *touch_fea = &itervar_map_->find(var)->second;

      // check if it is in the longest chain of the tree
      bool found = false;
      for (auto x : touch_fea->touch_feature) {
        if (innermost_buffers_->find(x.first) != innermost_buffers_->end()) {
          found = true;
          break;
        }
      }
      // if it is not in the longest chain of the tree, skip this subtree
      if (!found) return false;

      // length
      node->additional.push_back(static_cast<float>(length));
      // one hot annotation
      for (int i = 0; i < kNum; i++) {
        node->additional.push_back(static_cast<float>
                                   (i == ann_type));
      }
      // buffer access patten
      node->additional.push_back(static_cast<float>(touch_fea->topdown_product));
      for (auto x : touch_fea->touch_feature) {
        if (innermost_buffers_->find(x.first) == innermost_buffers_->end())
          continue;
        node->additional.push_back(static_cast<float>(x.second.count));
        node->additional.push_back(static_cast<float>(x.second.reuse));
      }
    }
    // add itervar as child
    node->children.push_back(std::make_shared<Tree>(var));

    root_stack_.back()->children.push_back(node);
    root_stack_.push_back(node);
    return true;
  }

  void ExitItervar_() {
    root_stack_.pop_back();
  }

  void EnterMem_(VarExpr buffer_var, Expr index) {
    if (itervar_map_ != nullptr)
      return;

    std::shared_ptr<Tree> node = std::make_shared<Tree>(buffer_var);
    IndexvarCollector collector;
    collector.Collect(index);

    for (const Variable *op : collector.vars)
      node->children.push_back(std::make_shared<Tree>(op->name_hint));

    for (auto iter = root_stack_.rbegin(); iter != root_stack_.rend(); iter++) {
      if (iter->get()->name == "for") {  // attach to nearest loop father node
        iter->get()->children.push_back(node);
        break;
      }
    }

    root_stack_.push_back(node);
  }

  void ExitMem_() {
    if (itervar_map_ != nullptr)
      return;

    root_stack_.pop_back();
  }

 private:
  std::deque<std::shared_ptr<Tree>> root_stack_;
  const std::unordered_map<VarExpr, ItervarFeature,
                           tvm::ExprHash, tvm::ExprEqual> *itervar_map_;
  const std::set<TouchedBuffer> *innermost_buffers_;
};

// serialize a tree
int DFSSerialize(std::shared_ptr<const Tree> root,
                 std::vector<std::vector<int>> *children,
                 std::vector<std::string> *names,
                 std::vector<std::vector<float>> *additionals) {
  std::vector<int> node_children;
  for (auto child : root->children) {
    int child_id = DFSSerialize(child, children, names, additionals);
    node_children.push_back(child_id);
  }

  int idx = static_cast<int>(children->size());
  children->push_back(node_children);
  names->push_back(root->name);
  additionals->push_back(root->additional);

  return idx;
}

/*!
 * \brief Get simplified AST
 *
 * \param stmt The IR to extract
 * \param add_stats whether add manual statics feature to the treenode
 * \param data Return buffer to store data
 *
 * \note: We serialize data and use ByteArray to return it. This method is faster.
 *
 * Data Format:
 * offset_child, offset_name, offset_additional, n_tree
 * offset_child: n_tree x int32, n1 x int32, n2 x int32, ...
 * offset_name: n_tree x int32, n1 x char, n2 x char, ...
 * offset_additional: n_tree x int32, n1 x float32, n2 x float32
 */
void GetSimplifiedAST(Stmt stmt, bool add_stats, std::vector<char> *data) {
  std::shared_ptr<Tree> root = std::make_shared<Tree>("root");

  ASTExtractor extractor;

  if (add_stats) {
    TouchExtractor touch_ext;

    // extract touch feature
    touch_ext.Analyze(stmt);

    // sort loop vars according to order
    std::vector<VarExpr> vars;
    for (auto kv : touch_ext.itervar_map) {
      vars.push_back(kv.first);
    }
    std::sort(vars.begin(), vars.end(), [&](const VarExpr &lhs, const VarExpr &rhs) -> bool {
      return touch_ext.itervar_map[lhs].order < touch_ext.itervar_map[rhs].order;
    });

    // find maximum depth of loop nests and the innermost buffers
    int max_depth = 0;
    std::set<std::string> added;
    std::set<TouchedBuffer> innermost_buffers;

    for (auto var : vars) {
      ItervarFeature &fea = touch_ext.itervar_map[var];
      max_depth = std::max(max_depth, fea.nest_level);
    }

    // mark inner most buffer
    for (auto iter = vars.rbegin(); iter != vars.rend(); iter++) {
      auto var = *iter;
      ItervarFeature &fea = touch_ext.itervar_map[var];
      if (fea.nest_level == max_depth) {
        for (auto kv : fea.touch_feature) {
          // FIXME(lmzheng): fix substr extraction
          std::string raw_name = kv.first.substr(0, kv.first.size() - 2);
          size_t pos = raw_name.find(".");
          if (pos < kv.first.size())
            raw_name = raw_name.substr(0, pos);

          if (added.find(raw_name) == added.end()) {
            innermost_buffers.insert(kv.first);
            added.insert(raw_name);
          }
        }
      }
    }

    extractor.Extract(stmt, root, &touch_ext.itervar_map, &innermost_buffers);
  } else {
    extractor.Extract(stmt, root, nullptr, nullptr);
  }

  // serialize tree structure for front end
  std::vector<std::vector<int>> children;
  std::vector<std::string> names;
  std::vector<std::vector<float>> additionals;
  DFSSerialize(root, &children, &names, &additionals);

  // calculate size
  int32_t n_tree = static_cast<int>(children.size());
  int32_t offset_child, offset_name, offset_additional;
  int32_t nbytes_child, nbytes_name, nbytes_add;
  int32_t total_size;

  nbytes_child = nbytes_name = nbytes_add = n_tree * sizeof(int32_t);
  for (int i = 0; i < n_tree; i++) {
    nbytes_child += children[i].size() * sizeof(int32_t);
    nbytes_name += names[i].size() * sizeof(char);
    nbytes_add += additionals[i].size() * sizeof(float);
  }

  offset_child = sizeof(int32_t) * 4;
  offset_name = offset_child + nbytes_child;
  offset_additional = offset_name + nbytes_name;
  total_size = offset_additional + nbytes_add;

  // serialize to bytes
  data->resize(static_cast<size_t>(total_size), 0);
  char *pdata = data->data();
  int32_t header[] = {n_tree, offset_child, offset_name, offset_additional};

  memcpy(pdata, header, sizeof(header));
  int32_t ct, num;

  ct = 0;
  for (int i = 0; i < n_tree; i++) {
    num = static_cast<int32_t>(children[i].size());
    memcpy(pdata + offset_child + sizeof(num) * i, &num, sizeof(num));
    memcpy(pdata + offset_child + sizeof(num) * n_tree + ct * sizeof(int32_t),
           children[i].data(), num * sizeof(int32_t));
    ct += num;
  }

  ct = 0;
  for (int i = 0; i < n_tree; i++) {
    num = static_cast<int32_t>(names[i].size());
    memcpy(pdata + offset_name + sizeof(num) * i, &num, sizeof(num));
    memcpy(pdata + offset_name + sizeof(num) * n_tree + ct * sizeof(int8_t),
           names[i].data(), num * sizeof(int8_t));
    ct += num;
  }

  ct = 0;
  for (int i = 0; i < n_tree; i++) {
    num = static_cast<int32_t>(additionals[i].size());
    memcpy(pdata + offset_additional + sizeof(num) * i, &num, sizeof(num));
    memcpy(pdata + offset_additional + sizeof(num) * n_tree + ct * sizeof(float),
           additionals[i].data(), num * sizeof(float));
    ct += num;
  }
}

TVM_REGISTER_API("autotvm.feature.GetSimplifiedAST")
.set_body([](tvm::TVMArgs args, tvm::TVMRetValue *ret) {
  Stmt stmt = args[0];
  bool add_statics = args[1];

  std::vector<char> data;
  GetSimplifiedAST(stmt, add_statics, &data);

  // pack return values
  TVMByteArray arr;
  arr.size = data.size();
  arr.data = data.data();
  *ret = arr;
});

}  // namespace autotvm
}  // namespace tvm
