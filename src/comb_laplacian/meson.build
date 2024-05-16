# py.extension_module(
#   '_claplacian',
#   'laplacian.cpp',
#   install: true,
#   include_directories: includes
# ) 

python_sources = [
  '__init__.py', 
  'filtration_cpu.py'
]

## Print install directory
message('Source install dir = '+py.get_install_dir())
py.install_sources(
  python_sources,
  subdir: pkg_name, 
  pure: false       # Will be installed next to binaries
)