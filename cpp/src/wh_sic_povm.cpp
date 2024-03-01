#include "wh_sic_povm.hpp"


G_Matrix::G_Matrix (const FiducialVector &vector) : dim_(vector.size())
{
  elements_ = c_vec(dim_*(dim_ + 1)/2, c_num(0.0, 0.0));
  calculate_elements_(vector);
}


size_t G_Matrix::index_ (const size_t k, const size_t l) const
{
  assert (l >= k);
  return l + dim_*k - k*(k + 1)/2;
}


c_num &G_Matrix::idx (const size_t k, const size_t l)
{
  return elements_[index_(k, l)];
}


c_num G_Matrix::idx (const size_t k, const size_t l) const
{
  return elements_[index_(k, l)];
}


c_num G_Matrix::calculate_one_element_ (const size_t k, const size_t l, const FiducialVector &v) const
{
  using namespace std;
  c_num G_kl (0.0, 0.0);
  for (size_t m=0; m<dim_; m++)
    G_kl += v[m] * conj(v[m+k]) * conj(v[m+l]) * v[m+k+l];
  return G_kl;
}


void G_Matrix::calculate_elements_ (const FiducialVector &v)
{
  for (size_t k=0; k<dim_; k++)
    for (size_t l=k; l<dim_; l++)
      idx(k, l) = calculate_one_element_(k, l, v);
}


void G_Matrix::update (const FiducialVector &v)
{
  assert (v.size() == dim_);
  calculate_elements_(v);
}


double G_Matrix::sum_of_elements_squared () const
{
  double s = 0;
  for (const c_num &e : elements_)
    s += std::norm(e);
  s *= 2;
  for (size_t k=0; k<dim_; k++)
    s -= std::norm(idx(k, k));
  return s;
}


double loss (const G_Matrix &g)
{
  return g.sum_of_elements_squared() - 2.0/(g.dim() + 1);
}


void numerical_gradient (const FiducialVector &v, FiducialVector &grad_v)
{
  assert ((v.size() == grad_v.size() ));
  const double eps = 1e-10;
  double a_grad_real, a_grad_imag;

  // create a copy of vector to calculate change under in loss under a small
  // change in each of the elements in v
  FiducialVector v2 (v);

  const double l = loss(G_Matrix(v));

  // vary real and imag part of each element and calculate partial derivative
  for (size_t i=0; i<v.size(); i++){
    // vary real part of v[i]
    v2[i] = c_num (v[i].real() + eps, v[i].imag());
    a_grad_real = (loss(v2) - l)/eps;

    // vary imaginary part of v[i]
    v2[i] = c_num (v[i].real(), v[i].imag() + eps);
    a_grad_imag = (loss(v2) - l)/eps;

    grad_v[i] = c_num (a_grad_real, a_grad_imag);
    v2[i] = v[i];
  }
}


void gradient (const FiducialVector &v, FiducialVector &grad_v)
{
  const G_Matrix g(v);
  gradient(g, v, grad_v);
}

void gradient (const G_Matrix &g, const FiducialVector &v, FiducialVector &grad_v)
{
  using namespace std;
  assert (v.size() == grad_v.size());

  const unsigned int d = v.size();

  double a_grad_real, a_grad_imag;
  c_num D_ikl;
  for (size_t i=0; i<d; i++){
    a_grad_real = 0.0;
    a_grad_imag = 0.0;
    for (size_t k=0; k<d; k++){
      for (size_t l=k+1; l<d; l++){
        D_ikl = v[i+k] * v[i+l] * conj(v[i+k+l])
          + conj(v[i-k+d]) * v[i-k+l+d] * conj(v[i+l])
          + conj(v[i-l+d]) * v[i+k-l+d] * conj(v[i+k])
          + conj(v[i-k-l+2*d]) * v[i-l+d] * v[i-k+d];
        a_grad_real += (g.idx(k, l)*D_ikl).real();

        D_ikl = v[i+k] * v[i+l] * conj(v[i+k+l])
          - conj(v[i-k+d]) * v[i-k+l+d] * conj(v[i+l])
          - conj(v[i-l+d]) * v[i+k-l+d] * conj(v[i+k])
          + conj(v[i-k-l+2*d]) * v[i-l+d] * v[i-k+d];
        a_grad_imag += (g.idx(k, l)*D_ikl).imag();
      }
    }
    a_grad_real *= 2;
    a_grad_imag *= 2;
    for (size_t k=0; k<d; k++){
      D_ikl = v[i+k] * v[i+k] * conj(v[i+2*k])
        + 2.0 * conj(v[i-k+d]) * v[i] * conj(v[i+k])
        + conj(v[i-2*k+2*d]) * v[i-k+d] * v[i-k+d];
      a_grad_real += (g.idx(k, k)*D_ikl).real();

      D_ikl = v[i+k] * v[i+k] * conj(v[i+2*k])
        - 2.0 * conj(v[i-k+d]) * v[i] * conj(v[i+k])
        + conj(v[i-2*k+2*d]) * v[i-k+d] * v[i-k+d];
      a_grad_imag += (g.idx(k, k)*D_ikl).imag();
    }
    grad_v[i] = c_num (2*a_grad_real, 2*a_grad_imag );
  }
}


void gradient_normalized (const G_Matrix &g, const FiducialVector &v, FiducialVector &grad_v)
{
  const unsigned int d { v.size() };

  // calculate the unnormalized gradient
  gradient(g, v, grad_v);
  double v_times_grad_over_n_sq { 0.0 };
  for (size_t i=0; i<d;i++)
    v_times_grad_over_n_sq += v[i].real()*grad_v[i].real() + v[i].imag()*grad_v[i].imag();
  const double n = v.norm();
  v_times_grad_over_n_sq *= 1/(n*n);

  // remove component along the direction of v
  for (size_t i=0; i<d; i++)
    grad_v[i] -= v[i]*v_times_grad_over_n_sq;
}
