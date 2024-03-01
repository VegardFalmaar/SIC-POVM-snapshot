#include "utils.hpp"


std::string to_string_3digit_zero_padded (const unsigned int n)
{
  using namespace std;
  const size_t num_dig = 3;
  string str = to_string(n);
  assert (str.length() <= num_dig);
  string z_pad = string(num_dig - min(num_dig, str.length()), '0') + str;
  return z_pad;
}


void SICPOVMResult::save_vector_to_file () const
{
  using namespace std;
  ofstream ofile {
    "output/" + to_string_3digit_zero_padded(dim()) + "_"
      + to_string(random_seed_) + "_"
      + to_string(vec_idx_) + "_fiducial.txt"
  };
  if (!ofile)
    throw runtime_error("Unable to open file to save fiducial");
  ofile << vector;
}


void SICPOVMResult::save_loss_to_file () const
{
  using namespace std;
  ofstream ofile {
    "output/loss/" + to_string_3digit_zero_padded(dim()) + "_"
      + to_string(random_seed_) + "_"
      + to_string(vec_idx_) + "_loss.csv"
  };
  if (!ofile)
    throw runtime_error("Unable to open file to save loss");
  for (size_t i=0; i<losses_.size(); i++)
    ofile << i*100 << ", " << setprecision(8) << losses_[i] << "\n";
}


void save_meta_data (
  const unsigned int dim,
  const unsigned int duration_ms,
  const unsigned int init_seed,
  const unsigned int max_num_seeds,
  const unsigned int num_threads,
  const bool solution_found
) {
  using namespace std;
  ofstream ofile { "output/" + to_string_3digit_zero_padded(dim) + "_data.txt" };
  if (!ofile)
    throw runtime_error("Unable to open file to save fiducial");

  ofile
    << "Initial seed:\n" << init_seed
    << "\n\nMax number of seeds tested:\n" << max_num_seeds
    << "\n\nNumber of threads:\n" << num_threads
    << "\n\nDuration (ms):\n" << duration_ms
    << "\n\nResult:\n" << ((solution_found) ? "Solution found" : "No solution found");
}
