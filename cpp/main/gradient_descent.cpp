/// @file

#include <chrono>
#include <omp.h>

#include "../src/utils.hpp"
#include "../src/vectors.hpp"


constexpr unsigned int DESIRED_NUM_THREADS = 17;
constexpr unsigned int MAX_NUM_SEEDS = 100;
constexpr unsigned int INIT_SEED = 227;
constexpr unsigned int MAX_ITER = 1e5;

bool save_loss_to_file (const unsigned int batch)
{
  // return (batch == 0);
  return false;
}

/*! \brief Loop through vector batch seeds and vector batches to find a
 * fiducial vector in a given dimension.
 */
void find_fiducial_vector (const unsigned int dim);

/*! \brief Loop through vector batches to find a fiducial vector given a
 * specific random seed.
 */
bool find_fiducial_specific_seed (
  const unsigned int dim,
  const unsigned int seed
);
bool find_fiducial_specific_seed_small_dim (
  const unsigned int dim,
  const unsigned int seed
);
bool find_fiducial_specific_seed_large_dim (
  const unsigned int dim,
  const unsigned int seed
);

/*! \brief Loop through initial vectors in a given batch.
 *
 * This function is called by all OpenMP threads in parallel.
 */
bool find_fiducial_specific_batch (
  const unsigned int dim,
  const unsigned int seed,
  const unsigned int batch,
  const LatinSquares &random_vector_gen
);

/*! \brief Use GD to transform a given initial vector to the one which
 * minimizes the G matrix loss.
 *
 * This function is called by all OpenMP threads in parallel.
 */
bool minimize_initial_vector (
  const unsigned int dim,
  const unsigned int seed,
  const unsigned int vector_index,
  FiducialVector v,
  const bool save_loss
);


int main ()
{
  // for (unsigned int dim=37; dim<40; dim++)
    // find_fiducial_vector(dim);
  find_fiducial_vector(42);

  return 0;
}


void find_fiducial_vector (const unsigned int dim)
{
  using namespace std;
  using time = chrono::high_resolution_clock;

  cout << "\nStarting dimension " << dim << ":\n" << flush;
  const time::time_point time_start = time::now();

  bool solution_found = false;

  for (size_t seed=INIT_SEED; seed<INIT_SEED + MAX_NUM_SEEDS; seed++){
    solution_found = find_fiducial_specific_seed(dim, seed);
    if (solution_found)
      break;
  }

  const time::time_point time_stop = time::now();
  const unsigned int duration_ms
    = chrono::duration_cast<chrono::milliseconds>(time_stop - time_start).count();
  cout << "Elapsed time: " << duration_ms << " ms = "
    << duration_ms / 1000 << " s = "
    << duration_ms / (double) 60000 << " mins" << endl;

  save_meta_data(
    dim,
    duration_ms,
    INIT_SEED,
    MAX_NUM_SEEDS,
    DESIRED_NUM_THREADS,
    solution_found
  );
}


bool find_fiducial_specific_seed (
  const unsigned int dim,
  const unsigned int seed
) {
  std::cout << "Seed " << seed << ": " << std::flush;
  bool solution_found;
  if (dim < 35)
    solution_found = find_fiducial_specific_seed_small_dim(dim, seed);
  else
    solution_found = find_fiducial_specific_seed_large_dim(dim, seed);

  if (solution_found)
    std::cout << "Fiducial found\n" << std::flush;
  else
    std::cout << "No fiducial\n" << std::flush;

  return solution_found;
}


bool find_fiducial_specific_seed_small_dim (
  const unsigned int dim,
  const unsigned int seed
) {
  LatinSquares random_vector_gen(dim);
  random_vector_gen.distribute_bins(seed);

  omp_set_num_threads(DESIRED_NUM_THREADS);

  const int padding = 64/sizeof(bool);
  bool solution_found_threads[DESIRED_NUM_THREADS*padding];
  for (unsigned int i=0; i<DESIRED_NUM_THREADS*padding; i++)
    solution_found_threads[i] = false;

  bool solution_found = false;

  #pragma omp parallel
  {
  const unsigned int num_threads = omp_get_num_threads();
  for (unsigned int batch=0; batch<NUM_BINS/num_threads; batch++){
    solution_found_threads[omp_get_thread_num()*padding] = find_fiducial_specific_batch(
      dim, seed, batch, random_vector_gen
    );
    #pragma omp barrier
    #pragma omp master
    {
    for (unsigned int i=0; i<num_threads; i++)
      if (solution_found_threads[i*padding]){
        solution_found = true;
        break;
      }
    }
    #pragma omp barrier
    if (solution_found)
      break;
  }
  if (not solution_found){
    solution_found_threads[omp_get_thread_num()*padding] = find_fiducial_specific_batch(
      dim, seed, NUM_BINS/num_threads, random_vector_gen
    );
    #pragma omp barrier
    #pragma omp master
    {
    for (unsigned int i=0; i<num_threads; i++)
      if (solution_found_threads[i*padding]){
        solution_found = true;
        break;
      }
    }
  }
  }

  return solution_found;
}


bool find_fiducial_specific_seed_large_dim (
  const unsigned int dim,
  const unsigned int seed
) {
  unsigned int num_threads;
  LatinSquares random_vector_gen(dim);
  random_vector_gen.distribute_bins(seed);

  omp_set_num_threads(DESIRED_NUM_THREADS);

  const int padding = 64/sizeof(bool);
  bool solution_found_threads[DESIRED_NUM_THREADS*padding];
  for (unsigned int i=0; i<DESIRED_NUM_THREADS*padding; i++)
    solution_found_threads[i] = false;

  #pragma omp parallel
  {
  num_threads = omp_get_num_threads();

  // retrieve boolean flag to be set by this thread
  bool *sol_found = &(solution_found_threads[omp_get_thread_num()*padding]);

  #pragma omp for
  for (unsigned int vec=0; vec<NUM_BINS; vec++){
    const bool vec_is_fiducial = minimize_initial_vector(
      dim, seed, vec, random_vector_gen.generate_initial_vector(vec), false
    );
    // update boolean flag for this thread
    *sol_found = (*sol_found || vec_is_fiducial);
  }
  }

  // loop through array filled out by threads to see if one of them found a solution
  bool solution_found = false;
  for (unsigned int i=0; i<num_threads; i++)
    if (solution_found_threads[i*padding]){
      solution_found = true;
      break;
    }

  return solution_found;
}


bool find_fiducial_specific_batch (
  const unsigned int dim,
  const unsigned int seed,
  const unsigned int batch,
  const LatinSquares &random_vector_gen
) {
  const unsigned int num_threads = omp_get_num_threads();
  const unsigned int vec_start = batch*num_threads;
  const unsigned int vec_stop = std::min(NUM_BINS, (batch + 1)*num_threads);

  #pragma omp single
  {
  std::cout << vec_start << "-" << vec_stop - 1 << ", " << std::flush;
  }

  bool solution_found_thread = false;
  #pragma omp for nowait
  for (unsigned int vec=vec_start; vec<vec_stop; vec++){
    solution_found_thread = minimize_initial_vector(
      dim, seed, vec,
      random_vector_gen.generate_initial_vector(vec),
      save_loss_to_file(batch)
    );
  }

  return solution_found_thread;
}


bool minimize_initial_vector (
  const unsigned int dim,
  const unsigned int seed,
  const unsigned int vector_index,
  FiducialVector v,
  const bool save_loss
) {
  unsigned char step_size_reduced = 0;
  double step_size = 0.2, loss_;

  FiducialVector grad_v(dim);
  SICPOVMResult res(v, seed, vector_index);
  G_Matrix g(res.vector);
  double prev_loss = loss(g);
  bool solution_found = false;

  for (unsigned int iter=0; iter<MAX_ITER; iter+=100){
    for (int inner=0; inner<100; inner++){
      gradient_normalized(g, res.vector, grad_v);
      for (size_t i=0; i<dim; i++)
        res.vector[i] -= step_size*grad_v[i];
      g.update(res.vector);
    }
    res.vector.remove_complex_phase();
    res.vector.normalize();

    loss_ = loss(g);
    if (save_loss)
      res.append_loss(loss_);

    // break if change in loss is less than 0.1%
    if (loss_ > 0.999*prev_loss)
      break;
    prev_loss = loss_;

    // break if we have found a global minimum
    if (loss_ < 1e-15){
      solution_found = true;
      break;
    }

    // reduce step size as we approach minimum
    if ((step_size_reduced == 0) && (loss_ < 1e-13)){
      step_size *= 0.5;
      step_size_reduced = 1;
    }
    else if ((step_size_reduced == 1) && (loss_ < 1e-14)){
      step_size *= 0.5;
      step_size_reduced = 2;
    }
  }

  if (solution_found){
    assert (res.is_fiducial());
    res.save_vector_to_file();
  }

  if (save_loss)
    res.save_loss_to_file();

  return solution_found;
}
