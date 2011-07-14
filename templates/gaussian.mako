/*
 * Gaussian Mixture Model Clustering with CUDA
 *
 * Orginal Author: Andrew Pangborn
 * Department of Computer Engineering
 * Rochester Institute of Technology
 * 
 */

#define PI  3.1415926535897931
#define COVARIANCE_DYNAMIC_RANGE 1E6
#define MINVALUEFORMINUSLOG -1000.0


#  define CUT_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
   }

#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);            \


typedef struct return_component_container
{
  boost::python::object component;
  //pyublas::numpy_array<float> distance;
  float distance;
} ret_c_con_t;

ret_c_con_t ret;
  
//=== Data structure pointers ===

//CPU copies of events
float *fcs_data_by_event;
float *fcs_data_by_dimension;

//GPU copies of events
float* d_fcs_data_by_event;
float* d_fcs_data_by_dimension;

//Index lists for accessing subsets of events
//CPU
int* index_list;
//GPU
int* d_index_list;

//CPU copies of components
components_t components;
components_t saved_components;
components_t** scratch_component_arr; // for computing distances and merging
static int num_scratch_components = 0;

//CPU copies of eval data
float *component_memberships;
float *loglikelihoods;
float *temploglikelihoods;

//GPU copies of components
components_t temp_components;
components_t* d_components;

//GPU copies of eval data
float *d_component_memberships;
float *d_loglikelihoods;
float *d_temploglikelihoods;


//=================================
float *LOOKUP_TABLE;
int N_LOOKUP_SIZE = 12;


//AHC functions
void copy_component(components_t *dest, int c_dest, components_t *src, int c_src, int num_dimensions);
void add_components(components_t *components, int c1, int c2, components_t *temp_component, int num_dimensions);
float component_distance(components_t *components, int c1, int c2, components_t *temp_component, int num_dimensions);
//end AHC functions

//Copy functions to ensure CPU data structures are up to date
void copy_component_data_GPU_to_CPU(int num_components, int num_dimensions);
void copy_evals_data_GPU_to_CPU(int num_events, int num_components);

// Function prototypes
void writeCluster(FILE* f, components_t components, int c,  int num_dimensions);
void printCluster(components_t components, int c, int num_dimensions);
void invert_cpu(float* data, int actualsize, float* log_determinant);
int invert_matrix(float* a, int n, float* determinant);

//============ LUTLOG ==============
void do_table(int n,float *lookup_table)
{
  float numlog;
  int *const exp_ptr = ((int*)&numlog);
  int x = *exp_ptr;
  x = 0x00000000;
  x += 127 << 23;
  *exp_ptr = x;
  for(int i=0;i<pow((double) 2,(double) n);i++)
    {
      lookup_table[i]=log2(numlog);
      x+=1 << (23-n);
      *exp_ptr = x;
    }
}

void create_lut_log_table() {

  unsigned int tablesize = (unsigned int)pow(2.0, 12);
  LOOKUP_TABLE = (float*) malloc(tablesize*sizeof(float));
  do_table(N_LOOKUP_SIZE,LOOKUP_TABLE);
 
}


components_t* alloc_temp_component_on_CPU(int num_dimensions) {

  components_t* scratch_component = (components_t*)malloc(sizeof(components_t));

  scratch_component->N = (float*) malloc(sizeof(float));
  scratch_component->pi = (float*) malloc(sizeof(float));
  scratch_component->CP = (float*) malloc(sizeof(float));
  scratch_component->constant = (float*) malloc(sizeof(float));
  scratch_component->avgvar = (float*) malloc(sizeof(float));
  scratch_component->means = (float*) malloc(sizeof(float)*num_dimensions);
  scratch_component->R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);
  scratch_component->Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);

  return scratch_component;
}

void dealloc_temp_components_on_CPU() {

for(int i = 0; i<num_scratch_components; i++) {
  free(scratch_component_arr[i]->N);
  free(scratch_component_arr[i]->pi);
  free(scratch_component_arr[i]->CP);
  free(scratch_component_arr[i]->constant);
  free(scratch_component_arr[i]->avgvar);
  free(scratch_component_arr[i]->means);
  free(scratch_component_arr[i]->R);
  free(scratch_component_arr[i]->Rinv);
  }
  num_scratch_components = 0;

  return;
}

// ================== Seed Components function - to initialize the clusters  ================= :

//TODO: do we want events be passed from Python?
void seed_components(int num_dimensions, int num_components, int num_events) {

  //  printf("SEEDING\n");
  seed_components_launch(d_fcs_data_by_event, d_components, num_dimensions, num_components, num_events);
  CUT_CHECK_ERROR("SEED FAIL");
}


// ================== Event data allocation on CPU  ================= :
void alloc_events_on_CPU(pyublas::numpy_array<float> input_data, int num_events, int num_dimensions) {

  //printf("Alloc events on CPU\n");

  fcs_data_by_event = input_data.data();
  // Transpose the event data (allows coalesced access pattern in E-step kernel)
  // This has consecutive values being from the same dimension of the data 
  // (num_dimensions by num_events matrix)
  fcs_data_by_dimension  = (float*) malloc(sizeof(float)*num_events*num_dimensions);

  for(int e=0; e<num_events; e++) {
    for(int d=0; d<num_dimensions; d++) {
      fcs_data_by_dimension[d*num_events+e] = fcs_data_by_event[e*num_dimensions+d];
    }
  }
  return;
}

// == Allocate index list on the CPU ==
//void alloc_index_list_on_CPU(pyublas::numpy_array<int> input_index_list) {
void alloc_index_list_on_CPU(pyublas::numpy_vector<int> input_index_list) {

  //printf("Alloc index list on CPU\n");

  index_list = input_index_list.data().data();
  //index_list = input_index_list.data();
 
  return;
}

void alloc_events_from_index_on_CPU(pyublas::numpy_array<float> input_data, pyublas::numpy_array<int> indices, int num_indices, int num_dimensions) {

 
  fcs_data_by_event = (float*)malloc(num_indices*num_dimensions*sizeof(int));
  for(int i = 0; i<num_indices; i++) {
    for(int d = 0; d<num_dimensions; d++) {
      fcs_data_by_event[i*num_dimensions+d] = input_data[indices[i]*num_dimensions+d];
    }
  }


  fcs_data_by_dimension = (float*)malloc(num_indices*num_dimensions*sizeof(int));
  for(int e=0; e<num_indices; e++) {
    for(int d = 0; d<num_dimensions; d++) {
      fcs_data_by_dimension[d*num_indices+e] = fcs_data_by_event[e*num_dimensions+d];
      //printf("data: %f\n", fcs_data_by_dimension[d*num_indices+e]);
    }
  }
  

}


// ================== Event data allocation on GPU  ================= :

void alloc_events_on_GPU(int num_dimensions, int num_events) {
  //printf("Alloc events on GPU\n");
  int mem_size = num_dimensions*num_events*sizeof(float);
    
  // allocate device memory for FCS data
  CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_event, mem_size));
  CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_dimension, mem_size));

  return;
}

void alloc_index_list_on_GPU(int num_indices) {
  //printf("Alloc index list on GPU\n");
  int mem_size = num_indices*sizeof(int);
    
  // allocate device memory for index list
  CUDA_SAFE_CALL(cudaMalloc( (void**) &d_index_list, mem_size));

  return;
}

void alloc_events_from_index_on_GPU(int num_indices, int num_dimensions) {

  CUDA_SAFE_CALL(cudaMalloc((void**) &d_fcs_data_by_event, sizeof(float)*num_indices*num_dimensions));
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_fcs_data_by_dimension, sizeof(float)*num_indices*num_dimensions));

  CUT_CHECK_ERROR("Alloc events on GPU from index");
}

//hack hack..
void relink_components_on_CPU(pyublas::numpy_array<float> weights, pyublas::numpy_array<float> means, pyublas::numpy_array<float> covars) {
     components.pi = weights.data();
     components.means = means.data();
     components.R = covars.data();
}

// ================== Cluster data allocation on CPU  ================= :

void alloc_components_on_CPU(int original_num_components, int num_dimensions, pyublas::numpy_array<float> weights, pyublas::numpy_array<float> means, pyublas::numpy_array<float> covars, pyublas::numpy_array<float> comp_probs) {

  //components.pi = (float*) malloc(sizeof(float)*original_num_components);
  //components.means = (float*) malloc(sizeof(float)*num_dimensions*original_num_components);   
  //components.R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_components);
  components.pi = weights.data();
  components.means = means.data();
  components.R = covars.data();
  components.CP = comp_probs.data();
  
  // components.CP = (float*)malloc(sizeof(float)*original_num_components); //NEW LINE
  components.N = (float*) malloc(sizeof(float)*original_num_components);      
  components.constant = (float*) malloc(sizeof(float)*original_num_components);
  components.avgvar = (float*) malloc(sizeof(float)*original_num_components);
  components.Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_components);
 
  return;
}  

// ================== Cluster data allocation on GPU  ================= :
void alloc_components_on_GPU(int original_num_components, int num_dimensions) {

  // Setup the component data structures on device
  // First allocate structures on the host, CUDA malloc the arrays
  // Then CUDA malloc structures on the device and copy them over
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.N),sizeof(float)*original_num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.pi),sizeof(float)*original_num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.CP),sizeof(float)*original_num_components)); //NEW LINE
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.constant),sizeof(float)*original_num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.avgvar),sizeof(float)*original_num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.means),sizeof(float)*num_dimensions*original_num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.R),sizeof(float)*num_dimensions*num_dimensions*original_num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.Rinv),sizeof(float)*num_dimensions*num_dimensions*original_num_components));
   
  // Allocate a struct on the device 
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_components, sizeof(components_t)));
    
  // Copy Cluster data to device
  CUDA_SAFE_CALL(cudaMemcpy(d_components,&temp_components,sizeof(components_t),cudaMemcpyHostToDevice));

  return;
}

// ================= Eval data alloc on CPU and GPU =============== 

void alloc_evals_on_CPU(pyublas::numpy_array<float> component_mem_np_arr, pyublas::numpy_array<float> loglikelihoods_np_arr, int num_events, int num_components){
  component_memberships = component_mem_np_arr.data();
  loglikelihoods = loglikelihoods_np_arr.data();
  temploglikelihoods = (float*)malloc(sizeof(float)*num_events*num_components);
}

void alloc_evals_on_GPU(int num_events, int num_components){
  CUDA_SAFE_CALL(cudaMalloc((void**) &(d_component_memberships),sizeof(float)*num_events*num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(d_loglikelihoods),sizeof(float)*num_events));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(d_temploglikelihoods),sizeof(float)*num_events*num_components));
}

void copy_evals_CPU_to_GPU(int num_events, int num_components) {

  CUDA_SAFE_CALL(cudaMemcpy( d_loglikelihoods, loglikelihoods, sizeof(float)*num_events,cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL(cudaMemcpy( d_temploglikelihoods, temploglikelihoods, sizeof(float)*num_events*num_components,cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL(cudaMemcpy( d_component_memberships, component_memberships, sizeof(float)*num_events*num_components,cudaMemcpyHostToDevice) );

}

// ======================== Copy event data from CPU to GPU ================
void copy_event_data_CPU_to_GPU(int num_events, int num_dimensions) {

  //printf("Copy events to GPU\n");
  int mem_size = num_dimensions*num_events*sizeof(float);
  // copy FCS to device
  CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data_by_event, fcs_data_by_event, mem_size,cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data_by_dimension, fcs_data_by_dimension, mem_size,cudaMemcpyHostToDevice) );
  return;
}

// == Index list copy from CPU to GPU
void copy_index_list_data_CPU_to_GPU(int num_indices) {
  //printf("Copy indices to GPU\n");
  int mem_size = num_indices*sizeof(int);

  CUDA_SAFE_CALL(cudaMemcpy( d_index_list, index_list, mem_size,cudaMemcpyHostToDevice) );

  return;
}

// == Event copy from indices
void copy_events_from_index_CPU_to_GPU(int num_indices, int num_dimensions) {

  CUDA_SAFE_CALL(cudaMemcpy(d_fcs_data_by_dimension, fcs_data_by_dimension, sizeof(float)*num_indices*num_dimensions, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_fcs_data_by_event, fcs_data_by_event, sizeof(float)*num_indices*num_dimensions, cudaMemcpyHostToDevice));
  CUT_CHECK_ERROR("Copy events from CPU to GPU execution failed: ");   

  
}

// ======================== Copy component data from CPU to GPU ================
void copy_component_data_CPU_to_GPU(int num_components, int num_dimensions) {

   CUDA_SAFE_CALL(cudaMemcpy(temp_components.N, components.N, sizeof(float)*num_components,cudaMemcpyHostToDevice)); 
   CUDA_SAFE_CALL(cudaMemcpy(temp_components.pi, components.pi, sizeof(float)*num_components,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(temp_components.CP, components.CP, sizeof(float)*num_components,cudaMemcpyHostToDevice)); //NEW LINE
   CUDA_SAFE_CALL(cudaMemcpy(temp_components.constant, components.constant, sizeof(float)*num_components,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(temp_components.avgvar, components.avgvar, sizeof(float)*num_components,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(temp_components.means, components.means, sizeof(float)*num_dimensions*num_components,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(temp_components.R, components.R, sizeof(float)*num_dimensions*num_dimensions*num_components,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(temp_components.Rinv, components.Rinv, sizeof(float)*num_dimensions*num_dimensions*num_components,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(d_components,&temp_components,sizeof(components_t),cudaMemcpyHostToDevice));
   return;
}
// ======================== Copy component data from GPU to CPU ================
void copy_component_data_GPU_to_CPU(int num_components, int num_dimensions) {

  CUDA_SAFE_CALL(cudaMemcpy(&temp_components, d_components, sizeof(components_t),cudaMemcpyDeviceToHost));
  // copy all of the arrays from the structs
  CUDA_SAFE_CALL(cudaMemcpy(components.N, temp_components.N, sizeof(float)*num_components,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(components.pi, temp_components.pi, sizeof(float)*num_components,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(components.CP, temp_components.CP, sizeof(float)*num_components,cudaMemcpyDeviceToHost)); //NEW LINE
  CUDA_SAFE_CALL(cudaMemcpy(components.constant, temp_components.constant, sizeof(float)*num_components,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(components.avgvar, temp_components.avgvar, sizeof(float)*num_components,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(components.means, temp_components.means, sizeof(float)*num_dimensions*num_components,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(components.R, temp_components.R, sizeof(float)*num_dimensions*num_dimensions*num_components,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(components.Rinv, temp_components.Rinv, sizeof(float)*num_dimensions*num_dimensions*num_components,cudaMemcpyDeviceToHost));
  
  return;
}

// ======================== Copy eval data from GPU to CPU ================
void copy_evals_data_GPU_to_CPU(int num_events, int num_components){
  CUDA_SAFE_CALL(cudaMemcpy(component_memberships, d_component_memberships, sizeof(float)*num_events*num_components, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(loglikelihoods, d_loglikelihoods, sizeof(float)*num_events, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(temploglikelihoods, d_temploglikelihoods, sizeof(float)*num_events*num_components, cudaMemcpyDeviceToHost));
}

// ================== Event data dellocation on CPU  ================= :
void dealloc_events_on_CPU() {
  //free(fcs_data_by_event);
  free(fcs_data_by_dimension);
  return;
}

// Index list
void dealloc_index_list_on_CPU() {
  free(index_list);
  return;
}

// ================== Event data dellocation on GPU  ================= :
void dealloc_events_on_GPU() {
  CUDA_SAFE_CALL(cudaFree(d_fcs_data_by_event));
  CUDA_SAFE_CALL(cudaFree(d_fcs_data_by_dimension));
  return;
}

// Index list
void dealloc_index_list_on_GPU() {
  CUDA_SAFE_CALL(cudaFree(d_index_list));
  return;
}


// ==================== Cluster data deallocation on CPU =================  
void dealloc_components_on_CPU() {

  //free(components.pi);
  //free(components.means);
  //free(components.R);
  //free(components.CP);

  free(components.N);
  free(components.constant);
  free(components.avgvar);
  free(components.Rinv);
  return;
}

// ==================== Cluster data deallocation on GPU =================  
void dealloc_components_on_GPU() {
  CUDA_SAFE_CALL(cudaFree(temp_components.N));
  CUDA_SAFE_CALL(cudaFree(temp_components.pi));
  CUDA_SAFE_CALL(cudaFree(temp_components.CP));
  CUDA_SAFE_CALL(cudaFree(temp_components.constant));
  CUDA_SAFE_CALL(cudaFree(temp_components.avgvar));
  CUDA_SAFE_CALL(cudaFree(temp_components.means));
  CUDA_SAFE_CALL(cudaFree(temp_components.R));
  CUDA_SAFE_CALL(cudaFree(temp_components.Rinv));
  
  CUDA_SAFE_CALL(cudaFree(d_components));

  return;
}

// ==================== Eval data deallocation on CPU and GPU =================  
void dealloc_evals_on_CPU() {
  //free(component_memberships);
  //free(loglikelihoods);
  free(temploglikelihoods);
  return;
}

void dealloc_evals_on_GPU() {
  CUDA_SAFE_CALL(cudaFree(d_component_memberships));
  CUDA_SAFE_CALL(cudaFree(d_loglikelihoods));
  CUDA_SAFE_CALL(cudaFree(d_temploglikelihoods));
  return;
}

// Accessor functions for pi, means, covars 

pyublas::numpy_array<float> get_temp_component_pi(components_t* c){
  pyublas::numpy_array<float> ret = pyublas::numpy_array<float>(1);
  std::copy( c->pi, c->pi+1, ret.begin());
  return ret;
}

pyublas::numpy_array<float> get_temp_component_means(components_t* c, int D){
  pyublas::numpy_array<float> ret = pyublas::numpy_array<float>(D);
  std::copy( c->means, c->means+D, ret.begin());
  return ret;
}

pyublas::numpy_array<float> get_temp_component_covars(components_t* c, int D){
  pyublas::numpy_array<float> ret = pyublas::numpy_array<float>(D*D);
  std::copy( c->R, c->R+D*D, ret.begin());
  return ret;
}

//------------------------- AHC FUNCTIONS ----------------------------


//////////////////////////////////////////////////////////////////////////////////////
inline float lut_log (float val, float *lookup_table, int n)
{
  int *const     exp_ptr = ((int*)&val);
  int            x = *exp_ptr;
  const int      log_2 = ((x >> 23) & 255) - 127;
  x &= 0x7FFFFF;
  x = x >> (23-n);
  val=lookup_table[x];
  // printf("log2:%f\n", log_2);
  return ((val + log_2)* 0.69314718);

}


// sequentuially add logarithms
float Log_Add(float log_a, float log_b)
{
  float result;
  if(log_a < log_b)
    {
      float tmp = log_a;
      log_a = log_b;
      log_b = tmp;
    }
  //setting MIN...LOG so small, I don't even need to look
  if((log_b - log_a) <= MINVALUEFORMINUSLOG)
    {
      return log_a;
    }
  else
    {
      result = log_a + (float)(lut_log(1.0 + (double)(exp((double)(log_b - log_a))),LOOKUP_TABLE,N_LOOKUP_SIZE));
    }
  return result;
}


double Log_Likelihood(int DIM, int m, float *feature, float *means, float *covars, float CP)
{
  //float log_lkld;
  //float in_the_exp = 0.0, den = 0.0;
  double x,y=0,z;
  for(int i=0; i<DIM; i++)
    {
      x = feature[i]-means[DIM*m + i];
      z = covars[m*DIM*DIM + i*DIM+i];
      y += x*x/z;//+lut_log(2*3.141592654*z,LOOKUP_TABLE,N_LOOKUP_SIZE); LINE MODIFIED
      // printf("y = %f, feature[%d]  = %f, mean[%d] = %f \n", y, i, feature[i], i, means[i*m+i], m*DIM*DIM+i*DIM+i, covars[m*DIM*DIM + i*DIM+i]);
    }
  //printf("y  = %f, CP  = %f\n", y, CP);
  return((double)-0.5*(y+CP)); //LINE MODIFIED
}


float Log_Likelihood_KL(float *feature, int DIM, int gmm_M, float *gmm_weights, float *gmm_means, float *gmm_covars, float *gmm_CP)
{

  //float res = 0.0;
  float log_lkld= MINVALUEFORMINUSLOG ,aux;
  for(int i=0;i<gmm_M;i++)
    {
      // if(gmm_weights[i])
      // {
          aux = lut_log(gmm_weights[i],LOOKUP_TABLE,N_LOOKUP_SIZE) + Log_Likelihood(DIM, i, feature, gmm_means, gmm_covars, gmm_CP[i]);

          
          if(isnan(aux) || !finite(aux))
            {
              aux = MINVALUEFORMINUSLOG;
            }
          log_lkld = Log_Add(log_lkld, aux);
          //}
    }//for
  return log_lkld;
}


float compute_KL_distance(int DIM, int gmm1_M, int gmm2_M, pyublas::numpy_array<float> gmm1_weights_in, pyublas::numpy_array<float> gmm1_means_in, pyublas::numpy_array<float> gmm1_covars_in, pyublas::numpy_array<float> gmm1_CP_in, pyublas::numpy_array<float> gmm2_weights_in, pyublas::numpy_array<float> gmm2_means_in, pyublas::numpy_array<float> gmm2_covars_in, pyublas::numpy_array<float> gmm2_CP_in) {

  float aux;
  float log_g1,log_f1,log_g2,log_f2,f_log_g=0,f_log_f=0,g_log_f=0,g_log_g=0;
  float *point_a = new float[DIM];
  float *point_b = new float[DIM];

  float *gmm1_weights = gmm1_weights_in.data();
  float *gmm1_means = gmm1_means_in.data();
  float *gmm1_covars = gmm1_covars_in.data();
  float *gmm1_CP = gmm1_CP_in.data();
  float *gmm2_weights = gmm2_weights_in.data();
  float *gmm2_means = gmm2_means_in.data();
  float *gmm2_covars = gmm2_covars_in.data();
  float *gmm2_CP = gmm2_CP_in.data();

  
  for(int i=0;i<gmm1_M;i++)
    {
      log_g1=0;
      log_f1=0;
      for(int k=0;k<DIM;k++)
        {
          //Compute the two points
          for(int j=0;j<DIM;j++)
            {
              if(j==k){
                aux = sqrt(19.0)*sqrt(gmm1_covars[i*DIM*DIM + k*DIM+k]);
                point_a[j] = gmm1_means[i*DIM+j] + aux;
                point_b[j] = gmm1_means[i*DIM+j] - aux;
              }
              else{
                point_a[j] = gmm1_means[i*DIM+j];
                point_b[j] = gmm1_means[i*DIM+j];
              }
            }
          log_g1+=Log_Likelihood_KL(point_a, DIM, gmm2_M, gmm2_weights, gmm2_means, gmm2_covars, gmm2_CP)+Log_Likelihood_KL(point_b, DIM, gmm2_M, gmm2_weights, gmm2_means, gmm2_covars, gmm2_CP);
          log_f1+=Log_Likelihood_KL(point_a, DIM, gmm1_M, gmm1_weights, gmm1_means, gmm1_covars, gmm1_CP)+Log_Likelihood_KL(point_b, DIM, gmm1_M, gmm1_weights, gmm1_means, gmm1_covars, gmm1_CP);
        }

      f_log_g+=gmm1_weights[i]*log_g1;
      f_log_f+=gmm1_weights[i]*log_f1;

    }
  for(int i=0;i<gmm2_M;i++)
    {
      log_g2=0;
      log_f2=0;
      for(int k=0;k<DIM;k++)
        {
          for(int j=0;j<DIM;j++)
            {
              if(j==k){
                aux = sqrt(19.0)*sqrt(gmm2_covars[i*DIM*DIM + k*DIM+k]);
                point_a[j] = gmm2_means[i*DIM+j] + aux;
                point_b[j] = gmm2_means[i*DIM+j] - aux;
              }
              else{
                point_a[j] = gmm2_means[i*DIM+j];
                point_b[j] = gmm2_means[i*DIM+j];
              }
            }

          log_g2+=Log_Likelihood_KL(point_a, DIM, gmm2_M, gmm2_weights, gmm2_means, gmm2_covars, gmm2_CP)+Log_Likelihood_KL(point_b, DIM, gmm2_M, gmm2_weights, gmm2_means, gmm2_covars, gmm2_CP);
          log_f2+=Log_Likelihood_KL(point_a, DIM, gmm1_M, gmm1_weights, gmm1_means, gmm1_covars, gmm1_CP)+Log_Likelihood_KL(point_b, DIM, gmm1_M, gmm1_weights, gmm1_means, gmm1_covars, gmm1_CP);

          
        }
      g_log_g+=gmm2_weights[i]*log_g2;
      g_log_f+=gmm2_weights[i]*log_f2;

    }
  delete [] point_a;
  delete [] point_b;
  return 1.0/(2.0*DIM)*(f_log_f + g_log_g - f_log_g - g_log_f);
}




int compute_distance_rissanen(int c1, int c2, int num_dimensions) {
  // compute distance function between the 2 components

  components_t *new_component = alloc_temp_component_on_CPU(num_dimensions);

  float distance = component_distance(&components,c1,c2,new_component,num_dimensions);
  //printf("distance %d-%d: %f\n", c1, c2, distance);

  scratch_component_arr[num_scratch_components] = new_component;
  num_scratch_components++;
  
  ret.component = boost::python::object(boost::python::ptr(new_component));
  ret.distance = distance;

  return 0;

}

void merge_components(int min_c1, int min_c2, components_t *min_component, int num_components, int num_dimensions) {

  // Copy new combined component into the main group of components, compact them
  copy_component(&components,min_c1, min_component,0,num_dimensions);

  for(int i=min_c2; i < num_components-1; i++) {
  
    copy_component(&components,i,&components,i+1,num_dimensions);
  }

  //return boost::python::object(boost::python::ptr(components));
  //return boost::python::object(components);
  return;
}


float component_distance(components_t *components, int c1, int c2, components_t *temp_component, int num_dimensions) {
  // Add the components together, this updates pi,means,R,N and stores in temp_component

  add_components(components,c1,c2,temp_component,num_dimensions);
  //printf("%f, %f, %f, %f, %f, %f\n", components->N[c1], components->constant[c1], components->N[c2], components->constant[c2], temp_component->N[0], temp_component->constant[0]);
  return components->N[c1]*components->constant[c1] + components->N[c2]*components->constant[c2] - temp_component->N[0]*temp_component->constant[0];
  
}

void add_components(components_t *components, int c1, int c2, components_t *temp_component, int num_dimensions) {
  float wt1,wt2;
 
  wt1 = (components->N[c1]) / (components->N[c1] + components->N[c2]);
  wt2 = 1.0f - wt1;
    
  // Compute new weighted means
  for(int i=0; i<num_dimensions;i++) {
    temp_component->means[i] = wt1*components->means[c1*num_dimensions+i] + wt2*components->means[c2*num_dimensions+i];
  }
    
  // Compute new weighted covariance
  for(int i=0; i<num_dimensions; i++) {
    for(int j=i; j<num_dimensions; j++) {
      // Compute R contribution from component1
      temp_component->R[i*num_dimensions+j] = ((temp_component->means[i]-components->means[c1*num_dimensions+i])
                                             *(temp_component->means[j]-components->means[c1*num_dimensions+j])
                                             +components->R[c1*num_dimensions*num_dimensions+i*num_dimensions+j])*wt1;
      // Add R contribution from component2
      temp_component->R[i*num_dimensions+j] += ((temp_component->means[i]-components->means[c2*num_dimensions+i])
                                              *(temp_component->means[j]-components->means[c2*num_dimensions+j])
                                              +components->R[c2*num_dimensions*num_dimensions+i*num_dimensions+j])*wt2;
      // Because its symmetric...
      temp_component->R[j*num_dimensions+i] = temp_component->R[i*num_dimensions+j];
    }
  }
    
  // Compute pi
  temp_component->pi[0] = components->pi[c1] + components->pi[c2];
    
  // compute N
  temp_component->N[0] = components->N[c1] + components->N[c2];

  float log_determinant;
  // Copy R to Rinv matrix
  memcpy(temp_component->Rinv,temp_component->R,sizeof(float)*num_dimensions*num_dimensions);
  // Invert the matrix
  invert_cpu(temp_component->Rinv,num_dimensions,&log_determinant);
  // Compute the constant
  temp_component->constant[0] = (-num_dimensions)*0.5*logf(2*PI)-0.5*log_determinant;
    
  // avgvar same for all components
  temp_component->avgvar[0] = components->avgvar[0];
}

void copy_component(components_t *dest, int c_dest, components_t *src, int c_src, int num_dimensions) {
  dest->N[c_dest] = src->N[c_src];
  dest->pi[c_dest] = src->pi[c_src];
  dest->constant[c_dest] = src->constant[c_src];
  dest->avgvar[c_dest] = src->avgvar[c_src];
  memcpy(&(dest->means[c_dest*num_dimensions]),&(src->means[c_src*num_dimensions]),sizeof(float)*num_dimensions);
  memcpy(&(dest->R[c_dest*num_dimensions*num_dimensions]),&(src->R[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
  memcpy(&(dest->Rinv[c_dest*num_dimensions*num_dimensions]),&(src->Rinv[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
  // do we need to copy memberships?
}
//---------------- END AHC FUNCTIONS ----------------


void writeCluster(FILE* f, components_t components, int c, int num_dimensions) {
  fprintf(f,"Probability: %f\n", components.pi[c]);
  fprintf(f,"N: %f\n",components.N[c]);
  fprintf(f,"Means: ");
  for(int i=0; i<num_dimensions; i++){
    fprintf(f,"%.3f ",components.means[c*num_dimensions+i]);
  }
  fprintf(f,"\n");

  fprintf(f,"\nR Matrix:\n");
  for(int i=0; i<num_dimensions; i++) {
    for(int j=0; j<num_dimensions; j++) {
      fprintf(f,"%.3f ", components.R[c*num_dimensions*num_dimensions+i*num_dimensions+j]);
    }
    fprintf(f,"\n");
  }
  fflush(f);   
  /*
    fprintf(f,"\nR-inverse Matrix:\n");
    for(int i=0; i<num_dimensions; i++) {
    for(int j=0; j<num_dimensions; j++) {
    fprintf(f,"%.3f ", c->Rinv[i*num_dimensions+j]);
    }
    fprintf(f,"\n");
    } 
  */
}

void printCluster(components_t components, int c, int num_dimensions) {
  writeCluster(stdout,components,c,num_dimensions);
}


static float double_abs(float x);

static int 
ludcmp(float *a,int n,int *indx,float *d);

static void 
lubksb(float *a,int n,int *indx,float *b);

/*
 * Inverts a square matrix (stored as a 1D float array)
 * 
 * actualsize - the dimension of the matrix
 *
 * written by Mike Dinolfo 12/98
 * version 1.0
 */
void invert_cpu(float* data, int actualsize, float* log_determinant)  {
  int maxsize = actualsize;
  int n = actualsize;
  *log_determinant = 0.0;

  if (actualsize == 1) { // special case, dimensionality == 1
    *log_determinant = logf(data[0]);
    data[0] = 1.0 / data[0];
  } else if(actualsize >= 2) { // dimensionality >= 2
    for (int i=1; i < actualsize; i++) data[i] /= data[0]; // normalize row 0
    for (int i=1; i < actualsize; i++)  { 
      for (int j=i; j < actualsize; j++)  { // do a column of L
        float sum = 0.0;
        for (int k = 0; k < i; k++)  
          sum += data[j*maxsize+k] * data[k*maxsize+i];
        data[j*maxsize+i] -= sum;
      }
      if (i == actualsize-1) continue;
      for (int j=i+1; j < actualsize; j++)  {  // do a row of U
        float sum = 0.0;
        for (int k = 0; k < i; k++)
          sum += data[i*maxsize+k]*data[k*maxsize+j];
        data[i*maxsize+j] = 
          (data[i*maxsize+j]-sum) / data[i*maxsize+i];
      }
    }

    for(int i=0; i<actualsize; i++) {
      *log_determinant += log10(fabs(data[i*n+i]));
      //printf("log_determinant: %e\n",*log_determinant); 
    }
    //printf("\n\n");
    for ( int i = 0; i < actualsize; i++ )  // invert L
      for ( int j = i; j < actualsize; j++ )  {
        float x = 1.0;
        if ( i != j ) {
          x = 0.0;
          for ( int k = i; k < j; k++ ) 
            x -= data[j*maxsize+k]*data[k*maxsize+i];
        }
        data[j*maxsize+i] = x / data[j*maxsize+j];
      }
    for ( int i = 0; i < actualsize; i++ )   // invert U
      for ( int j = i; j < actualsize; j++ )  {
        if ( i == j ) continue;
        float sum = 0.0;
        for ( int k = i; k < j; k++ )
          sum += data[k*maxsize+j]*( (i==k) ? 1.0 : data[i*maxsize+k] );
        data[i*maxsize+j] = -sum;
      }
    for ( int i = 0; i < actualsize; i++ )   // final inversion
      for ( int j = 0; j < actualsize; j++ )  {
        float sum = 0.0;
        for ( int k = ((i>j)?i:j); k < actualsize; k++ )  
          sum += ((j==k)?1.0:data[j*maxsize+k])*data[k*maxsize+i];
        data[j*maxsize+i] = sum;
      }
  } else {
    printf("Error: Invalid dimensionality for invert(...)\n");
  }
}


/*
 * Another matrix inversion function
 * This was modified from the 'component' application by Charles A. Bouman
 */
int invert_matrix(float* a, int n, float* determinant) {
  int  i,j,f,g;
   
  float* y = (float*) malloc(sizeof(float)*n*n);
  float* col = (float*) malloc(sizeof(float)*n);
  int* indx = (int*) malloc(sizeof(int)*n);
  /*
    printf("\n\nR matrix before LU decomposition:\n");
    for(i=0; i<n; i++) {
    for(j=0; j<n; j++) {
    printf("%.2f ",a[i*n+j]);
    }
    printf("\n");
    }*/

  *determinant = 0.0;
  if(ludcmp(a,n,indx,determinant)) {
    printf("Determinant mantissa after LU decomposition: %f\n",*determinant);
    printf("\n\nR matrix after LU decomposition:\n");
    for(i=0; i<n; i++) {
      for(j=0; j<n; j++) {
        printf("%.2f ",a[i*n+j]);
      }
      printf("\n");
    }
       
    for(j=0; j<n; j++) {
      *determinant *= a[j*n+j];
    }
     
    printf("determinant: %E\n",*determinant);
     
    for(j=0; j<n; j++) {
      for(i=0; i<n; i++) col[i]=0.0;
      col[j]=1.0;
      lubksb(a,n,indx,col);
      for(i=0; i<n; i++) y[i*n+j]=col[i];
    }

    for(i=0; i<n; i++)
      for(j=0; j<n; j++) a[i*n+j]=y[i*n+j];
     
    printf("\n\nMatrix at end of clust_invert function:\n");
    for(f=0; f<n; f++) {
      for(g=0; g<n; g++) {
        printf("%.2f ",a[f*n+g]);
      }
      printf("\n");
    }
    free(y);
    free(col);
    free(indx);
    return(1);
  }
  else {
    *determinant = 0.0;
    free(y);
    free(col);
    free(indx);
    return(0);
  }
}

static float double_abs(float x)
{
  if(x<0) x = -x;
  return(x);
}

#define TINY 1.0e-20

static int
ludcmp(float *a,int n,int *indx,float *d)
{
  int i,imax=0,j,k;
  float big,dum,sum,temp;
  float *vv;

  vv= (float*) malloc(sizeof(float)*n);
   
  *d=1.0;
   
  for (i=0;i<n;i++)
    {
      big=0.0;
      for (j=0;j<n;j++)
        if ((temp=fabsf(a[i*n+j])) > big)
          big=temp;
      if (big == 0.0)
        return 0; /* Singular matrix  */
      vv[i]=1.0/big;
    }
       
   
  for (j=0;j<n;j++)
    {  
      for (i=0;i<j;i++)
        {
          sum=a[i*n+j];
          for (k=0;k<i;k++)
            sum -= a[i*n+k]*a[k*n+j];
          a[i*n+j]=sum;
        }
       
      /*
        int f,g;
        printf("\n\nMatrix After Step 1:\n");
        for(f=0; f<n; f++) {
        for(g=0; g<n; g++) {
        printf("%.2f ",a[f*n+g]);
        }
        printf("\n");
        }*/
       
      big=0.0;
      dum=0.0;
      for (i=j;i<n;i++)
        {
          sum=a[i*n+j];
          for (k=0;k<j;k++)
            sum -= a[i*n+k]*a[k*n+j];
          a[i*n+j]=sum;
          dum=vv[i]*fabsf(sum);
          //printf("sum: %f, dum: %f, big: %f\n",sum,dum,big);
          //printf("dum-big: %E\n",fabs(dum-big));
          if ( (dum-big) >= 0.0 || fabs(dum-big) < 1e-3)
            {
              big=dum;
              imax=i;
              //printf("imax: %d\n",imax);
            }
        }
       
      if (j != imax)
        {
          for (k=0;k<n;k++)
            {
              dum=a[imax*n+k];
              a[imax*n+k]=a[j*n+k];
              a[j*n+k]=dum;
            }
          *d = -(*d);
          vv[imax]=vv[j];
        }
      indx[j]=imax;
       
      /*
        printf("\n\nMatrix after %dth iteration of LU decomposition:\n",j);
        for(f=0; f<n; f++) {
        for(g=0; g<n; g++) {
        printf("%.2f ",a[f][g]);
        }
        printf("\n");
        }
        printf("imax: %d\n",imax);
      */


      /* Change made 3/27/98 for robustness */
      if ( (a[j*n+j]>=0)&&(a[j*n+j]<TINY) ) a[j*n+j]= TINY;
      if ( (a[j*n+j]<0)&&(a[j*n+j]>-TINY) ) a[j*n+j]= -TINY;

      if (j != n-1)
        {
          dum=1.0/(a[j*n+j]);
          for (i=j+1;i<n;i++)
            a[i*n+j] *= dum;
        }
    }
  free(vv);
  return(1);
}

#undef TINY

static void
lubksb(float *a,int n,int *indx,float *b)
{
  int i,ii,ip,j;
  float sum;

  ii = -1;
  for (i=0;i<n;i++)
    {
      ip=indx[i];
      sum=b[ip];
      b[ip]=b[i];
      if (ii >= 0)
        for (j=ii;j<i;j++)
          sum -= a[i*n+j]*b[j];
      else if (sum)
        ii=i;
      b[i]=sum;
    }
  for (i=n-1;i>=0;i--)
    {
      sum=b[i];
      for (j=i+1;j<n;j++)
        sum -= a[i*n+j]*b[j];
      b[i]=sum/a[i*n+i];
    }
}

