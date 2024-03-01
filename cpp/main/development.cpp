#include <iostream>

#include "../src/vectors.hpp"


int main ()
{
  using namespace std;
  FiducialVector v { c_num(1.0, 2.0), c_num(3.0, 4.0), c_num(5.0, 6.0) };
  cout << "The vector is v = \n" << v;

  return 0;
}
