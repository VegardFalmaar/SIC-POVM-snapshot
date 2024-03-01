#ifndef __TEST_HPP
#define __TEST_HPP

/// @file

#include <iostream>
#include <iomanip>
#include <cmath>

typedef bool (*test_func_t) ();
typedef struct {
  test_func_t function;
  char const *name;
} test_func_info_t;

// test/wh_sic_povm.cpp
bool test_loss_2d ();
bool test_loss_3d ();
bool test_loss_grad_analytic_2d ();
bool test_loss_grad_analytic_3d ();
bool test_loss_grad_analytic_agrees_with_numerical ();

// test/utils.cpp
bool test_to_string_3digit_zero_padded ();

// test/vectors.cpp
bool test_LatinSquares_bins_diversity ();
bool test_norm ();
bool test_vector_generator_throws_exception_for_bad_vector_idx ();

#endif
