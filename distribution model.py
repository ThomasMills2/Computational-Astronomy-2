import numpy as np
import matplotlib.pyplot as plt

#Central vs Satellite Galaxies

import numpy as np
import sys
import os.path
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt

hdulist = fits.open('GalCat-1.fits')
tbdata = hdulist[1].data

mass_halo = tbdata.field(0) 
RA = tbdata.field(1)
Dec = tbdata.field(2)
z_spec = tbdata.field(5)
halo_id = tbdata.field(7)
dx = tbdata.field(8)
dy = tbdata.field(9)
dz = tbdata.field(10)
N_sat = tbdata.field(11)
mag = tbdata.field(13)

#splitting data
mass_halo = tbdata.field(0)
data_array = np.array([mass_halo, RA, Dec, z_spec, halo_id, dx, dy, dz, N_sat, mag]).T
flag_RA = data_array[:,1]<(5*60)

flag_mass_halo_central = data_array[:,0]>0
flag_mass_halo_sattelite = data_array[:,0]<0

data_array_sub_cent = data_array[flag_mass_halo_central,:]
data_array_sub_sat = data_array[flag_mass_halo_sattelite,:]



print('Central galaxy:', flag_mass_halo_cental)
print('Satellite galaxy:', flag_mass_halo_sattelite)

plt.hist(data_array_sub_cent[:,3], bins=500, label='Central Galaxy')
plt.hist(data_array_sub_sat[:,3], bins=500, label='Satellite Galaxy')
plt.title('Redshift Distribution for Central and Satellite Galaxies')
plt.xlabel('Redshift, z')
plt.ylabel('Number of Galaxies')
plt.legend()
plt.show()


#second part
hdulist = fits.open('GalCat-1.fits')
tbdata = hdulist[1].data

mass_halo = tbdata.field(0)
RA = tbdata.field(1)
Dec = tbdata.field(2)
z_spec = tbdata.field(5)
halo_id = tbdata.field(7)
dx = tbdata.field(8)
dy = tbdata.field(9)
dz = tbdata.field(10)
N_sat = tbdata.field(11)
mag = tbdata.field(13)
 

data_array = np.array([mass_halo, RA, Dec, z_spec, halo_id, dx, dy, dz, N_sat, mag]).T
 

flag_mass_halo_cental= data_array[:,0]>0
flag_mass_halo_sattelite = data_array[:,0]<0
data_array_sub_cent=data_array[flag_mass_halo_cental,:]
data_array_sub_sat=data_array[flag_mass_halo_sattelite,:]
print('Central galaxy:', flag_mass_halo_cental)
print('Satellite galaxy:', flag_mass_halo_sattelite)
ZCent=(data_array_sub_cent[:,3])
ZSat=(data_array_sub_sat[:,3])
plt.hist(ZCent,bins=500)
plt.legend(['Central Galaxy','Satellite Galaxy'])
plt.xlabel('Redshift,z')
plt.ylabel('Number of Galaxies')
plt.title('Redshift Distribution for Central and Satellite Galaxies')

ax = plt.gca()
p = ax.patches
ZCentH=np.array(([patch.get_height() for patch in p]))

plt.close()

plt.hist(ZSat,bins=500)
ax1=plt.gca()
m=ax1.patches
plt.close()
ZSatH=np.array(([patch.get_height() for patch in m]))
ZFrac=ZSatH/(ZSatH+ZCentH)
x=np.linspace(0,1.4,500)
plt.plot(x,ZFrac,'-', linewidth=0.6)
plt.xlabel('Redshift, z')
plt.ylabel('Satellite Fraction, f_sat(z)')
plt.title('Satellite Fraction as a Function of Redshift')

# Halo Mass Function (Centrals only)

hdulist = fits.open('GalCat-1.fits')
tbdata = hdulist[1].data
mass_halo = tbdata.field(0)
RA = tbdata.field(1)
Dec = tbdata.field(2)
z_spec = tbdata.field(5)
halo_id = tbdata.field(7)
dx = tbdata.field(8)
dy = tbdata.field(9)
dz = tbdata.field(10)
N_sat = tbdata.field(11)
mag = tbdata.field(13)



data_array = np.array([mass_halo, RA, Dec, z_spec, halo_id, dx, dy, dz, N_sat, mag]).T

 
flag_mass_halo_cental= data_array[:,0]>0
flag_mass_halo_sattelite = data_array[:,0]<0
data_array_sub_cent=data_array[flag_mass_halo_cental,:]
data_array_sub_sat=data_array[flag_mass_halo_sattelite,:]
print('Central galaxy:', flag_mass_halo_cental)
print('Sattelite galaxy:', flag_mass_halo_sattelite)
ZCent=(data_array_sub_cent[:,3])
ZSat=(data_array_sub_sat[:,3])
plt.hist(ZCent,bins=500)
plt.legend(['Central Galaxy','Sattelite Galaxy'])
plt.xlabel('Redshift,z')
plt.ylabel('Number of Galaxies')
plt.title('Redshift Distribution for Central and Sattelite Galaxies')

ax = plt.gca()
p = ax.patches
ZCentH=np.array(([patch.get_height() for patch in p]))
plt.close()
plt.hist(ZSat,bins=500)
ax1=plt.gca()
m=ax1.patches
plt.close()
ZSatH=np.array(([patch.get_height() for patch in m]))
ZFrac=ZSatH/(ZSatH+ZCentH)
x=np.linspace(0,1.4,500)
plt.plot(x,ZFrac,'-', linewidth=0.6)
plt.xlabel('Redshift, z')
plt.ylabel('Sattelite Fraction, f_sat(z)')
plt.title('Sattelite Fraction as a Function of Redshift')
plt.close()
centmass = np.log((data_array_sub_cent[:,0]))
plt.hist(centmass, bins=500)
plt.xlabel('Mass of Haloes, (Solar Masses)')
plt.ylabel('Number of Haloes')
plt.title('Number of Haloes as Function of their Mass')

#Red shift evolution of the halo mass function

hdulist = fits.open('GalCat-1.fits')
tbdata = hdulist[1].data
mass_halo = tbdata.field(0)
RA = tbdata.field(1)
Dec = tbdata.field(2)
z_spec = tbdata.field(5)
halo_id = tbdata.field(7)
dx = tbdata.field(8)
dy = tbdata.field(9)
dz = tbdata.field(10)
N_sat = tbdata.field(11)
mag = tbdata.field(13)



data_array = np.array([mass_halo, RA, Dec, z_spec, halo_id, dx, dy, dz, N_sat, mag]).T

 
flag_mass_halo_cental= data_array[:,0]>0
flag_mass_halo_sattelite = data_array[:,0]<0
data_array_sub_cent=data_array[flag_mass_halo_cental,:]
data_array_sub_sat=data_array[flag_mass_halo_sattelite,:]
print('Central galaxy:', flag_mass_halo_cental)
print('Sattelite galaxy:', flag_mass_halo_sattelite)
ZCent=(data_array_sub_cent[:,3])
ZSat=(data_array_sub_sat[:,3])
plt.hist(ZCent,bins=500)
plt.legend(['Central Galaxy','Sattelite Galaxy'])
plt.xlabel('Redshift,z')
plt.ylabel('Number of Galaxies')
plt.title('Redshift Distribution for Central and Sattelite Galaxies')

ax = plt.gca()
p = ax.patches
ZCentH=np.array(([patch.get_height() for patch in p]))
plt.close()
plt.hist(ZSat,bins=500)
ax1=plt.gca()
m=ax1.patches
plt.close()
ZSatH=np.array(([patch.get_height() for patch in m]))
ZFrac=ZSatH/(ZSatH+ZCentH)
x=np.linspace(0,1.4,500)
plt.plot(x,ZFrac,'-', linewidth=0.6)
plt.xlabel('Redshift, z')
plt.ylabel('Sattelite Fraction, f_sat(z)')
plt.title('Sattelite Fraction as a Function of Redshift')
plt.close()
centmass = np.log((data_array_sub_cent[:,0]))
plt.hist(centmass, bins=500)
plt.xlabel('Mass of Haloes, (Solar Masses)')
plt.ylabel('Number of Haloes')
plt.title('Number of Haloes as Function of their Mass')

plt.close()

from scipy import stats
bin_edges = stats.mstats.mquantiles(ZCent, [0, 1/3, 2/3, 1])
n, bins, patches = plt.hist(ZCent, bins=bin_edges, log=True)
patches[0].set_fc('purple')
patches[1].set_fc('brown')
patches[2].set_fc('yellow')
plt.yticks([641152, 641153, 641154, 641155, 641156])
print(bin_edges)
plt.xlabel('Redshift, Z')
plt.ylabel('Number of Galaxies')
plt.title('Equi-Populated Histogram of Redshift Bands')

#Satellites vs halo mass

hdulist = fits.open('GalCat-1.fits')
tbdata = hdulist[1].data
mass_halo = tbdata.field(0)
RA = tbdata.field(1)
Dec = tbdata.field(2)
z_spec = tbdata.field(5)
halo_id = tbdata.field(7)
dx = tbdata.field(8)
dy = tbdata.field(9)
dz = tbdata.field(10)
N_sat = tbdata.field(11)
mag = tbdata.field(13)



data_array = np.array([mass_halo, RA, Dec, z_spec, halo_id, dx, dy, dz, N_sat, mag]).T

 
flag_mass_halo_cental= data_array[:,0]>0
flag_mass_halo_sattelite = data_array[:,0]<0
data_array_sub_cent=data_array[flag_mass_halo_cental,:]
data_array_sub_sat=data_array[flag_mass_halo_sattelite,:]
print('Central galaxy:', flag_mass_halo_cental)
print('Satellite galaxy:', flag_mass_halo_sattelite)
ZCent=(data_array_sub_cent[:,3])
ZSat=(data_array_sub_sat[:,3])
plt.hist(ZCent,bins=500)
plt.legend(['Central Galaxy','Satellite Galaxy'])
plt.xlabel('Redshift,z')
plt.ylabel('Number of Galaxies')
plt.title('Redshift Distribution for Central and Satellite Galaxies')

ax = plt.gca()
p = ax.patches
ZCentH=np.array(([patch.get_height() for patch in p]))
plt.close()
plt.hist(ZSat,bins=500)
ax1=plt.gca()
m=ax1.patches
plt.close()
ZSatH=np.array(([patch.get_height() for patch in m]))
ZFrac=ZSatH/(ZSatH+ZCentH)
x=np.linspace(0,1.4,500)
plt.plot(x,ZFrac,'-', linewidth=0.6)
plt.xlabel('Redshift, z')
plt.ylabel('Satellite Fraction, f_sat(z)')
plt.title('Satellite Fraction as a Function of Redshift')
plt.close()
centmass = np.log((data_array_sub_cent[:,0]))
#plt.hist(centmass, bins=500)
plt.xlabel('Mass of Haloes, (Solar Masses)')
plt.ylabel('Number of Haloes')
plt.title('Number of Haloes as Function of their Mass')

plt.close()
Nsat = data_array[:,8]
halo_mass = data_array[:,0]
plt.scatter(mass_halo, Nsat, marker='o')
plt.xlabel('Halo Mass, (solar masses)')
plt.ylabel('Number of Satellite Galaxies, N_sat')
plt.title('Number of Satellites against Halo Mass')



