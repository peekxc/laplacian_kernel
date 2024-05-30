import numpy as np 
# import memray

# with memray.Tracker("output_file.bin"):
x = np.zeros(5000, dtype = np.int64)
x[:500] = 1


y = np.zeros(5000, dtype = np.int64)

z = np.zeros(5000, dtype = np.int64)

ii = 0
for i in range(10000):
  ii += 1

Z = np.zeros(shape=(5000, 5000), dtype = np.int64)
Z[:] = -1

del y 

del x

