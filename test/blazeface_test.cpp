#include <gtest/gtest.h>

#include <filesystem>

#include "tvm_blazeface.h"

namespace fs = std::filesystem;

TEST(BlazeFaceTest, TestCanExecute) {
    auto model_path = fs::current_path() / "dummy.so";
    tvm_blazeface::TVM_Blazeface model(model_path);
    EXPECT_EQ(model.CanExecute(), false);
}
