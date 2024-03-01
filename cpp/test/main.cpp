#include "main.hpp"


bool run_tests () {
  using namespace std;

  bool test_passed;
  bool tests_passed = true;

  #define init_test_arr(NAME) {.function=NAME, .name=#NAME}
  test_func_info_t test_funcs[] {
    init_test_arr(test_loss_2d),
    init_test_arr(test_loss_3d),
    init_test_arr(test_loss_grad_analytic_2d),
    init_test_arr(test_loss_grad_analytic_3d),
    init_test_arr(test_loss_grad_analytic_agrees_with_numerical),

    init_test_arr(test_to_string_3digit_zero_padded),

    init_test_arr(test_LatinSquares_bins_diversity),
    init_test_arr(test_norm),
    init_test_arr(test_vector_generator_throws_exception_for_bad_vector_idx),
  };
  const int num_tests = sizeof(test_funcs) / sizeof(test_func_info_t);

  cout << "\nRunning " << num_tests << " test(s):\n\n" << left << setfill ('.');
  for (int i=0; i<num_tests; i++) {
    cout << setw (60) << test_funcs[i].name;
    test_passed = test_funcs[i].function();
    tests_passed = (tests_passed && test_passed);
    if (test_passed)
      cout << " passed\n";
    else
      cout << " failed x\n";
  }

  if (tests_passed)
    std::cout << "\nSuccess!" << std::endl;
  else
    std::cout << "\nFailed!" << std::endl;

  return tests_passed;
}

int main ()
{
  run_tests();

  return 0;
}
