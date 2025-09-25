import PyMieScatt as ps
import numpy as np

# Use np.trapz instead of ps.trapz
arr = [1, 2, 3]
print(np.trapz(arr))
