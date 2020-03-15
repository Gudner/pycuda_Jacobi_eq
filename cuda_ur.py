import sys
import pycuda.driver as cuda
import os
from pycuda import compiler, gpuarray, tools
from pycuda.compiler import SourceModule
import numpy as np

import time

import pycuda.autoinit

if (os.system("cl.exe")):
    os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.24.28314\bin\Hostx64\x64"
if (os.system("cl.exe")):
    raise RuntimeError("cl.exe still not found, path probably incorrect")

fout = open('res.txt', 'w')

ly = 1.0
lx = 1.0
h = 0.01
eps = 0.01
block_size = 32

n = int(lx / h) + 1
m = int(ly / h) + 1

u = np.zeros(n * m)
ub = np.zeros((n, m))

udev = gpuarray.empty(shape = u.shape, dtype = np.float32)
uprdev = gpuarray.empty(shape = u.shape, dtype = np.float32)

for i in range(n):
    ub[i][0] =  50.0 * i * h * (1.0 - i  * h)
    ub[i][n - 1] = 50.0 * i * h* (1.0 - i * h)

for i in range(n):
    ub[0][i] = 0.0
    ub [m - 1][i]= 50.0 * i * h * (1.0 - i * h* i * h)

#for i in range(n):
#    for j in range(m):
#        fout.write("ub[" + str(i) + "][" + str(j) + "] = " + str(ub[i][j]) + "\n")

for i in range(n):
    for j in range(m):
        u[i * n + j] = ub[i][j]

count = 0

udev = gpuarray.to_gpu(u)

kernel_code_template = """
__global__ void JacobiKernel(float *u, float *upr, int n, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    __syncthreads();
    if((i < (n-1)) && (j < (m-1)))
    {
        upr[i * m + j] = 0.25f * (u[(i - 1) * m + j] + u[(i + 1 )* m + j] + u[i* m + j - 1] + u[i* m + j + 1]);

    }
    __syncthreads();
    if((i < (n-1)) && (j < (m-1)))
        u[i * m + j] = upr[i * m + j];
}
"""

kernel_code = kernel_code_template % {
    'BLOCK_SIZE': block_size,
    }

mod = compiler.SourceModule(kernel_code)

Jacobi = mod.get_function("JacobiKernel")

while count != 10000:
    Jacobi(udev.gpudata, uprdev.gpudata, np.uint32(n), np.uint32(m), grid = (n // block_size + 1, n // block_size + 1), block = (block_size, block_size, 1))
    cuda.Context.synchronize()
    count += 1

udev.get(u)
for i in range(n):
    for j in range(m):
        fout.write("u[" + str(i) + "][" + str(j) + "] = " + str(u[i * n + j]) + "\n")

fout.close()

input('Press ENTER to exit\n')
