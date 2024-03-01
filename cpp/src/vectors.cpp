#include "vectors.hpp"


LatinSquares::LatinSquares (const unsigned int dim)
  : dim_ { dim }, num_params_ { 2*dim - 2 },
  bins_ { std::vector<unsigned int> (NUM_BINS*num_params_, 0) } {}


void LatinSquares::distribute_bins (const unsigned int seed)
{
  unsigned int range_indices[NUM_BINS];
  for (size_t i=0; i<NUM_BINS; i++)
    range_indices[i] = i;

  std::srand(seed);
  for (size_t param=0; param<num_params_; param++){
    std::random_shuffle(range_indices, range_indices + NUM_BINS);
    for (size_t i=0; i<NUM_BINS; i++)
      bins_[param*NUM_BINS + i] = range_indices[i];
  }
}


FiducialVector LatinSquares::generate_initial_vector (const unsigned int vector_idx) const
{
  if (vector_idx >= NUM_BINS)
    throw std::invalid_argument(
      "Vector index " + std::to_string(vector_idx)
      + " out of range for Latin Square with " + std::to_string(NUM_BINS)
      + " bins. \nCall distribute_bins(seed) to redistribute bins,"
      + " and start from 0."
    );

  std::vector<double> angles(num_params_, 0.0);
  for (size_t i=0; i<num_params_; i++)
    angles[i] = generate_number_from_bin_(bins_[i*NUM_BINS + vector_idx]);

  FiducialVector v = transform_spher_to_eucl(angles);
  assert (abs(v.norm_squared() - 1.0) < 1e-13);

  return v;
}


FiducialVector transform_spher_to_eucl(const std::vector<double> angles)
{
  const unsigned int dim = angles.size()/2 + 1;
  FiducialVector v(dim);

  double re, im;
  v[0] = c_num(cos(angles[0]), 0.0);
  double c = sin(angles[0]);
  for (size_t i=1; i<dim - 1; i++){
    re = c*cos(angles[2*i - 1]);
    c *= sin(angles[2*i - 1]);
    im = c*cos(angles[2*i]);
    c *= sin(angles[2*i]);
    v[i] = c_num(re, im);
  }
  re = c*cos(angles[2*dim - 3]);
  im = c*sin(angles[2*dim - 3]);
  v[dim - 1] = c_num(re, im);

  return v;
}


double LatinSquares::generate_number_from_bin_ (const unsigned int bin) const
{
  const double bin_size = 2.0/NUM_BINS;
  const double lower_bound = bin*bin_size - 1.0;
  const double c = lower_bound + 0.5*bin_size;
  const double x = acos(c);
  return x;
}


FiducialVector::FiducialVector (const unsigned int dim)
  : dim_(dim), values_(c_vec(dim, c_num(0, 0))) {}

FiducialVector::FiducialVector (std::initializer_list<c_num> values)
  : dim_(values.size()), values_(values) {}

FiducialVector::FiducialVector (c_vec values)
  : dim_(values.size()), values_(values) {}


c_num &FiducialVector::operator[] (const int idx)
{
  return values_[idx % dim_];
}


c_num FiducialVector::operator[] (const int idx) const
{
  return values_[idx % dim_];
}


std::ostream &operator<< (std::ostream &os, const FiducialVector &v)
{
  std::string pm;
  for (const auto &a : v.values_){
    pm = (a.imag() >= 0) ? "+" : "-";
    os << std::setprecision(16) << a.real() << pm << fabs(a.imag()) << "i\n";
  }
  return os;
}


double FiducialVector::norm_squared () const
{
  double n = 0.0;
  for (const auto &c : values_)
    n += std::norm(c);
  return n;
}


void FiducialVector::normalize ()
{
  const double n = norm ();
  for (auto &c : values_)
    c /= n;
}


void FiducialVector::remove_complex_phase ()
{
  c_num a0 = values_[0];
  const double phase = std::arg(a0);
  const c_num phase_factor (cos(phase), -sin(phase));
  for (auto &c : values_)
    c *= phase_factor;
}
