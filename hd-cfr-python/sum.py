import numpy as np

l_nom = np.array([1*0.8,2,3*1.2]) * 2
l_den = np.array([0.8,1,1.2]) * 2
nom = l_nom.sum()
den = l_den.sum()
print(f"nom: {nom}")
print(f"den: {den}")
print(nom / den)
