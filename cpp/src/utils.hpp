#ifndef __UTILS_HPP
#define __UTILS_HPP

/// @file

#include <iostream>
#include <fstream>

#include "vectors.hpp"
#include "wh_sic_povm.hpp"

/// Convert an integer to string of length 3 with preceding padding of 0's.
std::string to_string_3digit_zero_padded (const unsigned int n);

/*! \brief Container to hold a FiducialVector and other information about the
 * solution and computation, along with functionality for saving the result to file.
 */
class SICPOVMResult {
  private:
    unsigned int random_seed_;
    unsigned int vec_idx_;
    std::vector<double> losses_;

  public:
    FiducialVector vector;

    SICPOVMResult (FiducialVector v, unsigned int random_seed, unsigned int vector_idx) :
      random_seed_ { random_seed },
      vec_idx_ { vector_idx },
      vector { v } {};

    unsigned int dim () const { return vector.size(); };
    void append_loss (const double l) { losses_.push_back(l); };
    void save_vector_to_file () const;
    void save_loss_to_file () const;
    bool is_fiducial () const { return (loss(G_Matrix(vector)) < 2e-15); }
};

void save_meta_data (
  const unsigned int dim,
  const unsigned int duration_ms,
  const unsigned int init_seed,
  const unsigned int max_num_seeds,
  const unsigned int num_threads,
  const bool solution_found
);

#endif
