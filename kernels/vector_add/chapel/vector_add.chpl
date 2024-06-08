// Build with: chpl vector_add.chpl
//   Get detailed generated code with chpl --savec tmp vector_add.chpl

// Run with ./vector_add
// Can change vector size with ./vector_add --n 1000

use GpuDiagnostics;

config const n = 1000;

proc vecAdd(A, B, ref C)
{
    //C = A + B;
    foreach i in A.domain do
        C[i] = A[i] + B[i];
}

proc vecAddGpu(A, B, ref C)
{
    on here.gpus[0] {
        // Print information about memory transfers and kernel launches
        startVerboseGpu();
        // Must allocate GPU memory inside the 'on' statement
        var A_d: [A.domain] real = A;
        var B_d: [B.domain] real = B;
        var C_d: [C.domain] real = noinit;

        @assertOnGpu
        foreach i in A.domain do
            C_d[i] = A_d[i] + B_d[i];

        C = C_d;
        stopVerboseGpu();
    }
}

proc main() {
    var A: [0..<n] real;
    var B: [0..<n] real;
    var C: [0..<n] real;

    A = 1.0;

    for i in B.domain do
        B[i] = i;

    //vecAdd(A, B, C);
    vecAddGpu(A, B, C);


    writeln("C = ",C[0]," ",C[1]," ",C[2]);


}
