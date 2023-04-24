import matplotlib.pyplot as plt
import numpy as np

iter_max = 100
x = np.arange(iter_max)

plt.subplot(1,2,1)
plt.plot(x,np.cos(x))
plt.xlabel('t/$\eta$')
plt.ylabel('|m|(t)')
plt.xscale('log')
plt.subplot(1,2,2)
plt.plot(x,np.sin(x))
plt.xlabel('t/$\eta$')
plt.ylabel('$\mathcal{L}(t)$')
plt.xscale('log')
plt.yscale('log')
plt.savefig('test_image.png')