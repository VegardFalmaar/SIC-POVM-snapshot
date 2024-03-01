#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include "diver.hpp"
#include "../src/wh_sic_povm.hpp"
#include "../src/vectors.hpp"

// Number of derived quantities to output
const int nDerived = 0;
// Number of parameters that are to be treated as discrete
const int nDiscrete = 0;
// Indices of discrete parameters, Fortran style, i.e. starting at 1!!
const int discrete[] = {0};
// Split the population evenly amongst discrete parameters and evolve separately
const bool partitionDiscrete = false;
// Use current vector for mutation
const bool current = false;
// Use exponential crossover
const bool expon = false;
// Use self-adaptive choices for rand/1/bin parameters as per Brest et al 2006
const bool jDE = true;
// Use self-adaptive rand-to-best/1/bin parameters; based on Brest et al 2006
const bool lambdajDE = true;
// Threshold for gen-level convergence: smoothed fractional improvement in the mean population value
const double convthresh = 1.e-3;
// Population at which node is partitioned in binary space partitioning for posterior
const double maxNodePop = 1.9;
// Input tolerance in log-evidence
const double Ztolerance = 0.1;
// Restart from a previous run
const bool resume = false;
// Calculate approximate log evidence and posterior weightings
const bool doBayesian = false;
// Path to save samples, resume files, etc
const char path[] = "output_diver/diver";
// Maximum number of civilisations
const int maxciv = 100;
// Maximum number of generations per civilisation
const int maxgen = 100;
// Size of the array indicating scale factors
const int nF = 1;
// Scale factor(s). Note that this must be entered as an array.
const double F[nF] = {0.6};
// Crossover factor
const double Cr = 0.9;
// Mixing factor between best and rand/current
const double lambda = 0.8;
// Boundary constraint: 1=brick wall, 2=random re-initialization, 3=reflection
const int bndry = 3;
// Number of steps to smooth over when checking convergence
const int convsteps = 10;
// Weed out duplicate vectors within a single generation
const bool removeDuplicates = true;
// Save progress every savecount generations
const int savecount = 1;
// Write output .raw and .sam (if nDerived != 0) files
const bool outputSamples = false;
// Initialisation strategy: 0=one shot, 1=n-shot, 2=n-shot with error if no valid vectors found.
const int init_pop_strategy = 2;
// Recalculate any trial vector whose fitness is above max_acceptable_value
const bool discard_unfit_points = false;
// Maximum number of times to try to find a valid vector for each slot in the initial population.
const int max_init_attempts = 10000;
// Maximum fitness to accept for the initial generation if init_population_strategy > 0, or any generation if discard_unfit_points = true.
const double max_acceptable_val = 1e6;
// base seed for random number generation; non-positive or absent means seed from the system clock
const int seed = 1;
// Output verbosity: 0=only error messages, 1=basic info, 2=civ-level info, 3+=population info
const int verbose = 1;
void *context = nullptr;


void find_fiducial_vector (const unsigned int dim);


int main(int argc, char** argv)
{
  find_fiducial_vector(4);
}


FiducialVector expand_params_to_normalized_vector (const double params[], const int param_dim)
{
  std::vector<double> angles(param_dim);
  for (int i=0; i<param_dim; i++)
    angles[i] = acos(params[i]);
  return transform_spher_to_eucl(angles);
}


// function to be minimized.
double objective (double params[], const int param_dim, int &fcall, bool &quit,
    const bool validvector, void*& context)
{
  double result;
  fcall += 1;

  if (not validvector){
    result = std::numeric_limits<double>::max();
    return result;
  }

  FiducialVector v = expand_params_to_normalized_vector(params, param_dim);
  G_Matrix g(v);
  result = loss(g);
  quit = false;
  return result;
}


void find_fiducial_vector (const unsigned int dim)
{
  // Dimensionality of the parameter space
  // last parameter removed since norm of last c-number is given by norm(v) = 1
  const int nPar = 2*dim - 2;
  const int NP = 10*nPar;
  double lowerbounds[nPar];
  double upperbounds[nPar];
  for (int i=0; i<nPar; i++){
    lowerbounds[i] = -1.0;
    upperbounds[i] = 1.0;
  }

  // master call to Diver
  // Note that prior, maxNodePop and Ztolerance are ignored if doBayesian = false
  // prior is set to nullptr
  cdiver(objective, nPar, lowerbounds, upperbounds, path, nDerived, nDiscrete,
      discrete, partitionDiscrete, maxciv, maxgen, NP, nF, F, Cr, lambda,
      current, expon, bndry, jDE, lambdajDE, convthresh, convsteps,
      removeDuplicates, doBayesian, nullptr, maxNodePop, Ztolerance,
      savecount, resume, outputSamples, init_pop_strategy,
      discard_unfit_points, max_init_attempts, max_acceptable_val, seed,
      context, verbose);
}
