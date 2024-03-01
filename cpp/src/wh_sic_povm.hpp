#ifndef __WH_SIC_POVM_HPP
#define __WH_SIC_POVM_HPP

/// @file

#include <cassert>

#include "types.hpp"
#include "vectors.hpp"

/*! \brief The matrix G associated with the loss to be minimized when
 * determining fiducial vectors in the Weyl-Heisenberg group.
 */
class G_Matrix
{
  private:
    const unsigned int dim_;
    c_vec elements_;

    size_t index_ (const size_t k, const size_t l) const;
    c_num calculate_one_element_ (const size_t k, const size_t l, const FiducialVector &v) const;
    void calculate_elements_ (const FiducialVector &v);

  public:
    G_Matrix (const FiducialVector &vector);

    /// Recompute all elements of the matrix corresponding to an updated fiducial vector.
    void update (const FiducialVector &vector);

    /// Retrieve by reference the element in row k and column l.
    c_num &idx (const size_t k, const size_t l);

    /// Retrieve by value the element in row k and column l.
    c_num idx (const size_t k, const size_t l) const;

    /// Return the sum of all |G_kl|^2 for elements G_kl in matrix.
    double sum_of_elements_squared () const;

    /// The dimensionality of the complex vector to which the matrix belongs.
    unsigned int dim () const { return dim_; }
};

/// Compute the loss associated with the given instance of G_Matrix.
double loss (const G_Matrix &g);

/// Compute a numerical approximation of the gradient of the G_Matrix loss.
void numerical_gradient (const FiducialVector &v, FiducialVector &grad_v);

/// Compute the gradient of the G_Matrix loss using an exact expression.
void gradient (const FiducialVector &v, FiducialVector &grad_v);

/// Overload gradient to allow passing a G_Matrix for avoiding duplicate computations
void gradient (const G_Matrix &g, const FiducialVector &v, FiducialVector &grad_v);

/// Calculate the gradient of the loss of the normalized input vector
void gradient_normalized (const G_Matrix &g, const FiducialVector &v, FiducialVector &grad_v);

#endif
