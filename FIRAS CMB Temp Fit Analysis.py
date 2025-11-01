import numpy as np
import matplotlib.pyplot as plt

#Plot and basic analysis
data = np.loadtxt('FIRAS.txt')

x = data[:,0]
y = data[:,1]
error = data[:,3]

plt.plot(x,y, label='Spectral Radiance, B_v')
plt.xlabel('Frequency, f, (cm^-1)')
plt.ylabel('FIRAS monopole spectrum, B_v, (MJy/sr)')
plt.title('Spectral Radiance Data vs Frequency')
plt.errorbar(x, y, yerr=error, fmt=' ', label='Error')
plt.legend()
plt.grid()
plt.show()

print(np.mean(y))
print(np.std(y))

#Theoretical black body comparison
data = np.loadtxt('FIRAS.txt')

T = 2.725
k_B = 1.380649*10**-23
h = 6.62607015*10**-34
c = 299792458
v = data[:,0]*c*100
residuals = (5,9,15,4,19,-30,-30,-10,32,4,-2,13,-22,8,8,-21,9,12,11,-29,-46,58,6,-6,6,-17,6,26,-12,-19,8,7,14,-33,6,26,-26,-6,8,26,57,-116,-432)

B_v = (2*v**2/c**2)*(h*v/np.exp(h*v/(k_B*T)-1))

plt.plot(B_v)
plt.xlabel('Frequecy, f, (Hz)')
plt.ylabel('Flux, \u03D5, (kJy/sr)')
plt.title('Theoretical Black Body Curve')
plt.show()

data = np.loadtxt('FIRAS.txt')

T = 2.725
k_B = 1.380649*10**-23
h = 6.62607015*10**-34
c = 299792458
v = data[:,0]*c*100
residuals = (5,9,15,4,19,-30,-30,-10,32,4,-2,13,-22,8,8,-21,9,12,11,-29,-46,58,6,-6,6,-17,6,26,-12,-19,8,7,14,-33,6,26,-26,-6,8,26,57,-116,-432)

B_v = (2*v**2/c**2)*(h*v/np.exp(h*v/(k_B*T)-1))

plt.plot(B_v)
plt.plot(residuals, 'o')
plt.xlabel('Frequecy, f, (Hz)')
plt.ylabel('Flux, \u03D5, (kJy/sr)')
plt.title('Theoretical Black Body Curve with Residuals')
plt.show()

data = np.loadtxt('FIRAS.txt')

T = 2.725
k_B = 1.380649*10**-23
h = 6.62607015*10**-34
c = 299792458
v = data[:,0]*c*100
residuals = (5,9,15,4,19,-30,-30,-10,32,4,-2,13,-22,8,8,-21,9,12,11,-29,-46,58,6,-6,6,-17,6,26,-12,-19,8,7,14,-33,6,26,-26,-6,8,26,57,-116,-432)


def black_body(v, T):
    return (2*v**2/c**2)*(h*v/(np.exp(h*v/k_B*T)-1))

sigma_T = 0.001
bb_curve = black_body(v, T)
bb_curve_upper = black_body(v, T + sigma_T)
bb_curve_lower = black_body(v, T - sigma_T)

fig, axs = plt.subplots(3, 1, sharex=True)

axs[0].plot(v, bb_curve, label='T = 2.725 K')
axs[1].plot(v, bb_curve_upper, label='T=2.726 K', color='red')
axs[2].plot(v, bb_curve_lower, label='T=2.724 K', color='green')

fig.text(0.06, 0.5,'Flux, \u03D5, (kJy/sr)', ha='center', va='center', rotation='vertical')
fig.suptitle('Theoretical Black Body Curve at T = 2.725 K \u00B1 1\u03C3')
axs[0].legend()
axs[1].legend()
axs[2].legend()

plt.xlabel('Frequecy, f, (Hz)')
plt.show()

#Chi-square fitting

import numpy as np
from scipy.optimize import minimize

def planck(v, T):
    h = 6.62607004e-34
    c = 299792458
    k = 1.38064852e-23
    return 2 * h * v**3 / c**2 / (np.exp(h * v / k / T) - 1)

data = np.loadtxt('FIRAS.txt')
nu = data[:, 0] 
spec = data[: 1]
err = data[:, 3]
residual = data[: 2]

def chi2(T):
    return np.sum(residual ** 2)

result = minimize(chi2, 2.725)

print("Best-fitting temerature: %.6f K" % result.x[0])
print("Minimum chi-squared: %.6f" % result.fun)

cov = np.diag(err**2)

print("Covariance matrix:")
print(cov)
