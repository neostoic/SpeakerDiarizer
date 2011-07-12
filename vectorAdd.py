import numpy
import numpy.linalg as la
import asp.codegen.templating.template as AspTemplate
import asp.jit.asp_module as aspm
from codepy.cgen import *
from codepy.cuda import CudaModule
from codepy.cgen.cuda import CudaGlobal

#c_main_tpl = AspTemplate.Template(filename="templates/basic.mako")
c_main_tpl = AspTemplate.Template(filename="templates/vectorAdd_main.mako")
cu_kern_tpl = AspTemplate.Template(filename="templates/vectorAdd_kernel.mako")
c_main_rend = c_main_tpl.render()
cu_kern_rend = cu_kern_tpl.render()

host = aspm.ASPModule()
# remember, must specify function name when using a string
host.add_header('stdio.h')
host.add_header('cutil_inline.h')
global_vars = ['float* h_A;\n'
	'float* h_B;\n'
	'float* h_C;\n'
	'float* d_A;\n'
	'float* d_B;\n'
	'float* d_C;\n'
	'bool noprompt = false;\n']
for s in global_vars: host.add_to_preamble(s)
randominit_s = """void RandomInit(float* data, int n) {   for (int i = 0; i < n; ++i) data[i] = rand() / (float)RAND_MAX; }"""
cleanup_s = """void Cleanup() {
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_C) cudaFree(d_C);
    if (h_A) free(h_A);
    if (h_B) free(h_B);
    if (h_C) free(h_C);
    cutilSafeCall( cudaThreadExit() );
    return;}"""
host.add_function(c_main_rend, fname="main")
host.module.mod_body.append(Line(cleanup_s))
host.module.mod_body.insert(0, FunctionDeclaration(Value('void', "Cleanup"),[]))
host.module.mod_body.append(Line(randominit_s))
host.module.mod_body.insert(0, FunctionDeclaration(Value('void', "RandomInit"),[Pointer(POD(numpy.float32,'data')),POD(numpy.int32,'n')]))

cuda_mod = CudaModule(host.module)
cuda_mod.add_to_preamble([Include('cuda.h')])

cuda_mod.add_to_module([Line(cu_kern_rend)])

launch_statements = [   'int threadsPerBlock = 256;'
                        'int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;'
                        'VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);']
launch_func = FunctionBody(  
                FunctionDeclaration(Value('void', 'launch_VecAdd'),
                                [   Pointer(Value('float', 'd_A')), 
                                    Pointer(Value('float', 'd_B')), 
                                    Pointer(Value('float', 'd_C')), 
                                    Value('int', 'N')  ]),
                Block([Statement(s) for s in launch_statements]) )

cuda_mod.add_function(launch_func)

import codepy.toolchain
nvcc_toolchain = codepy.toolchain.guess_nvcc_toolchain()
host.toolchain.add_library("cutils",['/home/henry/NVIDIA_GPU_Computing_SDK/C/common/inc','/home/henry/NVIDIA_GPU_Computing_SDK/C/shared/inc'],[],[])
nvcc_toolchain.add_library("cutils",['home/henry/NVIDIA_GPU_Computing_SDK/C/common/inc','home/henry/NVIDIA_GPU_Computing_SDK/C/shared/inc'],[],[])

compiled_module = cuda_mod.compile(host.toolchain, nvcc_toolchain, debug=True)

compiled_module.main()
