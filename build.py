import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext

def build(setup_kwargs):
    """
    This function is required by Poetry to build the package with CUDA extensions.
    """
    setup_kwargs.update({
        'ext_modules': [
            make_cuda_ext(
                name='knn_cuda',
                module='unitraj.models.mtr.ops.knn',
                sources=[
                    'src/knn.cpp',
                    'src/knn_gpu.cu',
                    'src/knn_api.cpp',
                ],
            ),
            make_cuda_ext(
                name='attention_cuda',
                module='unitraj.models.mtr.ops.attention',
                sources=[
                    'src/attention_api.cpp',
                    'src/attention_func_v2.cpp',
                    'src/attention_func.cpp',
                    'src/attention_value_computation_kernel_v2.cu',
                    'src/attention_value_computation_kernel.cu',
                    'src/attention_weight_computation_kernel_v2.cu',
                    'src/attention_weight_computation_kernel.cu',
                ],
            ),
        ],
        'cmdclass': {
            'build_ext': BuildExtension
        }
    })

    return setup_kwargs
