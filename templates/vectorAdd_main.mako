// Host code
int main()
{
    printf("Vector addition test\n");
    int N = 50000;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float*)malloc(size);
    if (h_A == 0) Cleanup();
    h_B = (float*)malloc(size);
    if (h_B == 0) Cleanup();
    h_C = (float*)malloc(size);
    if (h_C == 0) Cleanup();
    
    // Initialize input vectors
    RandomInit(h_A, N);
    RandomInit(h_B, N);

    // Allocate vectors in device memory
    cutilSafeCall( cudaMalloc((void**)&d_A, size) );
    cutilSafeCall( cudaMalloc((void**)&d_B, size) );
    cutilSafeCall( cudaMalloc((void**)&d_C, size) );

    // Copy vectors from host memory to device memory
    cutilSafeCall( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );
    cutilSafeCall( cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) );

    printf("Invoking kernel...\n");
    // Invoke kernel
    //int threadsPerBlock = 256;
    //int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    //VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    launch_VecAdd(d_A, d_B, d_C, N);
    cutilCheckMsg("kernel launch failure");
    cutilSafeCall( cudaThreadSynchronize() );

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cutilSafeCall( cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost) );
    
    // Verify result
    int i;
    for (i = 0; i < N; ++i) {
        float sum = h_A[i] + h_B[i];
        if (fabs(h_C[i] - sum) > 1e-5)
            break;
    }
    printf("%s \n", (i == N) ? "PASSED" : "FAILED");
    
    Cleanup();

    return 0;
}


