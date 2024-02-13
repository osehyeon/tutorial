#include "src/main.h"
#include <gtest/gtest.h>

TEST(MainTest, AddFunction) {
    EXPECT_EQ(Add(2, 2), 4);
    EXPECT_EQ(Add(-1, 1), 0);
    EXPECT_EQ(Add(-3, -3), -6);
}