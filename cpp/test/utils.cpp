#include "../src/utils.hpp"


bool test_to_string_3digit_zero_padded ()
{
  bool passed = true;

  if (to_string_3digit_zero_padded(2) != "002")
    passed = false;

  if (to_string_3digit_zero_padded(9) != "009")
    passed = false;

  if (to_string_3digit_zero_padded(10) != "010")
    passed = false;

  if (to_string_3digit_zero_padded(31) != "031")
    passed = false;

  if (to_string_3digit_zero_padded(100) != "100")
    passed = false;

  if (to_string_3digit_zero_padded(972) != "972")
    passed = false;

  return passed;
}
