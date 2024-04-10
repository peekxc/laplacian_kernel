
import urllib.request

laplacian_url = "https://raw.githubusercontent.com/peekxc/laplacian_kernel/main/laplacian.cu"
f = urllib.request.urlopen(laplacian_url)
LP_cu = f.read()
LP_cu.decode("utf-8").replace("\\n", "\n")