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

#include "tensorflow/core/grappler/optimizers/tfg_optimizer_hook.h"

#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/ir/tf_op_wrapper.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace tfg {
// This is a MLIR TFG "test pass", it will just rename all the nodes with the
// suffix "_visited".
class TestPass : public PassWrapper<TestPass, OperationPass<GraphOp>> {
  StringRef getArgument() const override { return "grappler-hook-test-pass"; }
  void runOnOperation() override {
    GraphOp graph = getOperation();
    for (TFOp op : graph.getOps()) op.setName(op.name() + "_visited");
  }
};
}  // namespace tfg
}  // namespace mlir

namespace tensorflow {
namespace grappler {
namespace {

TEST(TFGOptimizerTest, TestCustomPipeline) {
  // Build a simple graph with two nodes, the test pass will rename them.
  Scope s = Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});
  Output b = ops::Const(s.WithOpName("b"), 1.0f, {10, 10});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  EXPECT_EQ("a", item.graph.node(0).name());
  EXPECT_EQ("b", item.graph.node(1).name());

  // This is testing that we can invoke an arbitrary pipeline, here the pass
  // registered above.
  mlir::tfg::TFGGrapplerOptimizer optimizer(
      [](mlir::PassManager &mgr) {
        mgr.addNestedPass<mlir::tfg::GraphOp>(
            std::make_unique<mlir::tfg::TestPass>());
      },
      /*num_tfg_threads=*/0);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  TF_ASSERT_OK(status);
  EXPECT_EQ("a_visited", output.node(0).name());
  EXPECT_EQ("b_visited", output.node(1).name());
}

TEST(TFGOptimizerTest, TestCustomPipelineName) {
  // Test printing the name of a custom pipeline.
  mlir::tfg::TFGGrapplerOptimizer optimizer(
      [](mlir::PassManager &mgr) {
        mgr.addNestedPass<mlir::tfg::GraphOp>(
            std::make_unique<mlir::tfg::TestPass>());
      },
      /*num_tfg_threads=*/0);
  EXPECT_EQ(optimizer.name(),
            "tfg_optimizer{tfg.graph(grappler-hook-test-pass)}");
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
