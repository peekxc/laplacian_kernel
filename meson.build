project('laplacian', 'cpp')

py = import('python').find_installation(pure: false)

py.extension_module(
  'laplacian',
  'laplacian.cu',
  install: true,
)