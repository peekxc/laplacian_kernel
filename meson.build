project('comb_laplacian', 'cpp')

pkg_name = meson.project_name()
py = import('python').find_installation(pure: false)
includes = []

## Include directories
incdir_numpy = run_command(py, ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'], check : true).stdout().strip()
incdir_pybind11 = run_command(py, ['-c', 'import os; os.chdir(".."); import pybind11; print(pybind11.get_include())'], check : true).stdout().strip()

## Header includes
includes += include_directories(incdir_numpy)
includes += include_directories(incdir_pybind11)

## Compile the package directory
subdir('src' / pkg_name)