#include "../src/vectors.hpp"


const double tol = 1e-15;


bool test_LatinSquares_bins_diversity ()
{
  for (unsigned int dim=2; dim<4; dim++){
    LatinSquares ls(dim);
    ls.distribute_bins(0);
    const std::vector<unsigned int> bins = ls.bins();
    for (unsigned int coeff=0; coeff<2*dim; coeff++){
      std::vector<char> bins_used(NUM_BINS, 0);
      for (unsigned int bin_idx=0; bin_idx<NUM_BINS; bin_idx++){
        const unsigned int bin = bins[coeff*NUM_BINS + bin_idx];
        if (bins_used[bin] == 1)
          return false;
        bins_used[bin] = 1;
      }
    }
  }
  return true;
}


bool test_norm ()
{
  bool passed = true;

  FiducialVector v { c_num(1, 0), c_num(1, 0) };
  if (fabs(v.norm() - sqrt(2)) > tol)
    passed = false;

  FiducialVector v2 { c_num(1, 1) };
  if (fabs(v2.norm() - sqrt(2)) > tol)
    passed = false;

  FiducialVector v3 { c_num(1, 1), c_num(-3, 4) };
  if (fabs(v3.norm() - 3*sqrt(3)) > tol)
    passed = false;

  return passed;
}


bool test_vector_generator_throws_exception_for_bad_vector_idx ()
{
  bool passed = true;

  LatinSquares ls(2);
  ls.distribute_bins(0);

  bool exception_raised = false;
  try { ls.generate_initial_vector(-1); }
  catch (std::invalid_argument &e) { exception_raised = true; }
  passed = (passed && exception_raised);

  exception_raised = false;
  try { ls.generate_initial_vector(NUM_BINS); }
  catch (std::invalid_argument &e) { exception_raised = true; }
  passed = (passed && exception_raised);

  exception_raised = false;
  try { ls.generate_initial_vector(NUM_BINS + 1); }
  catch (std::invalid_argument &e) { exception_raised = true; }
  passed = (passed && exception_raised);

  return passed;
}
