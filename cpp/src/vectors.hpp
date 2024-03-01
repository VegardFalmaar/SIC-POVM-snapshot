/// @file

#ifndef __VECTORS_HPP
#define __VECTORS_HPP

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <initializer_list>
#include <cassert>

#include "types.hpp"

const unsigned int NUM_BINS = 100;

/*! \brief A complex vector with functionality required for determining a
 * fiducial vector in arbitrary dimensions.
 */
class FiducialVector {
  private:
    unsigned int dim_;
    c_vec values_;

  public:
    /// Constructor where elements of vector default to c-number 0.0 + 0.0i
    FiducialVector (const unsigned int dimension);

    /*! \brief Constructor accepting an initializer list of values
     *
     * Usage: FiducialVector v { c_num(1, -1), c_num(2, -2), c_num(3, -3) };
     */
    FiducialVector (std::initializer_list<c_num> values);

    /// Constructor accepting a vector of complex numbers
    FiducialVector (c_vec values);

    /// Index elements by reference
    c_num &operator[] (const int idx);

    /// Index elements by value without the possibility to change state of vector
    c_num operator[] (const int idx) const;

    double norm_squared () const;
    double norm () const { return sqrt(norm_squared()); };

    /// Rescale all elements by norm such that new norm equals 1
    void normalize ();

    /// Multiply all elements with phase factor such that phase of first element is 0
    void remove_complex_phase ();
    unsigned int size () const { return dim_; }

    /*! \brief Overload << operator to let instance of the class to be passed
     * to stream e.g. for printing
     *
     * Vector is added to ostream object with one element per line on the form
     * a+bi, and ends in a newline.
     */
    friend std::ostream &operator<< (std::ostream &os, const FiducialVector &v);
};


class LatinSquares {
  private:
    const unsigned int dim_;
    const unsigned int num_params_;
    std::vector<unsigned int> bins_;

    double generate_number_from_bin_ (const unsigned int bin) const;

  public:
    LatinSquares (const unsigned int dim);
    void distribute_bins (const unsigned int seed);
    std::vector<unsigned int> bins () const { return bins_; };
    FiducialVector generate_initial_vector (const unsigned int vector_idx) const;
    unsigned int dim () const { return dim_; };
};

/// Transform an array of angles in spherical coordinates to a complex vector
FiducialVector transform_spher_to_eucl(const std::vector<double> angles);

#endif
