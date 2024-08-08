from typing import List
import numpy as np
import jinja2
import pyopencl as cl

def render(ctx: cl.Context, clbuild_options: List[str], src: List[str], dtype: np.dtype):

    '''
    Renders OpenCL source code with Jinja2 template engine.

    Parameters
    ----------
    ctx : pyopencl.Context
        OpenCL context.
    clbuild_options : list of str
        OpenCL build options.
    src : str
        OpenCL source code.
    dtype : numpy.dtype
        Data type of the arrays. Must be 
        either np.complex64 or np.complex128.
    
    Returns
    -------
    program : pyopencl.Program
        OpenCL program.
    '''

    render_ctx = {
            np.complex64: {'T':{'float': 'float'}},
            np.complex128: {'T':{'float': 'double'}}
        }.get(dtype.type)

    cl_source = jinja2.Template(src).render(**render_ctx)

    program = cl.Program(ctx, cl_source).build(
        options=clbuild_options)

    return program