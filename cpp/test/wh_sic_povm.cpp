#include "../src/wh_sic_povm.hpp"
#include "../src/vectors.hpp"
#include "main.hpp"


const double tol = 1e-15;


bool test_loss_2d ()
{
  bool passed = true;
  double expected;

  FiducialVector v { {
    c_num (1.0, 0.0),
    c_num (0.0, 0.0),
  } };
  expected = 1.0/3;
  if (fabs(loss(v) - expected) > tol)
    passed = false;

  v[0] = c_num (1.0, 0.0);
  v[1] = c_num (1.0, 0.0);
  expected = 46.0/3;
  if (fabs(loss(v) - expected) > tol)
    passed = false;

  v[0] = c_num (2.0, 0.0);
  v[1] = c_num (0.0, 1.0);
  expected = 1441.0/3;
  if (fabs(loss(v) - expected) > tol)
    passed = false;

  return passed;
}


bool test_loss_3d ()
{
  FiducialVector v { {
    c_num (0.1, 0.0),
    c_num (0.0, 0.2),
    c_num (0.3, 0.0),
  } };
  if (fabs(loss(v) - (-0.49977912)) > tol)
    return false;
  return true;
}


bool test_loss_grad_analytic_2d ()
{
  FiducialVector v { {
    c_num (0.5, -0.2),
    c_num (0.1, 0.7),
  } };
  FiducialVector v_grad { {
    c_num (0.0, 0.0),
    c_num (0.0, 0.0),
  } };
  gradient(v, v_grad);
  if (fabs(v_grad[0] - c_num (2.0998504, -0.5654416)) > tol)
    return false;
  if (fabs(v_grad[1] - c_num (0.5134352, 2.2215536)) > tol)
    return false;
  return true;
}


bool test_loss_grad_analytic_3d ()
{
  const double tol = 1e-14;
  FiducialVector v { {
    c_num (0.5, -0.2),
    c_num (0.1, 0.7),
    c_num (-0.6, -0.3),
  } };
  FiducialVector v_grad { {
    c_num (0.0, 0.0),
    c_num (0.0, 0.0),
    c_num (0.0, 0.0),
  } };
  gradient(v, v_grad);
  if (fabs(v_grad[0] - c_num (7.572512, -4.576256)) > tol)
    return false;
  if (fabs(v_grad[1] - c_num (0.3900928, 9.037936)) > tol)
    return false;
  if (fabs(v_grad[2] - c_num (-7.9629408, -4.219632)) > tol)
    return false;
  return true;
}


bool test_loss_grad_analytic_agrees_with_numerical ()
{
  const double tol = 1e-5;
  c_vec elements { {
    c_num (0.5, -0.2),
  } };
  double factor = 1.0;
  for (size_t dim=2; dim<10; dim++){
    FiducialVector v_grad_a(dim), v_grad_n(dim);
    factor *= -1.0;
    elements.push_back( c_num (factor*0.2*(1.0 + 1.0/dim), -factor*0.1*(1.0 + 1.0/dim)) );
    FiducialVector v(elements);
    v_grad_a = c_vec (dim, c_num(0.0, 0.0));
    v_grad_n = c_vec (dim, c_num(0.0, 0.0));
    gradient(v, v_grad_a);
    numerical_gradient(v, v_grad_n);
    for (size_t i=0; i<dim; i++)
      if (fabs(v_grad_a[i] - v_grad_n[i]) > tol)
        return false;
  }
  return true;
}
