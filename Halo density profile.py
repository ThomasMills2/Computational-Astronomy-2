import numpy as np
import matplotlib.pyplot as plt

#Dark matter halo density profile

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
ZCent=(data_array_sub_cent[:,3])
ZSat=(data_array_sub_sat[:,3])

flag_halo_id = (data_array_sub_sat[:,4]==2)
data_array_sub_2=(data_array_sub_sat[flag_halo_id,:])
dx=(data_array_sub_2[:,5])
dr=abs(dx)
#print(dr)
n,bins,patches=plt.hist(dr, bins=50)
#print(flag_halo_id)
plt.xlabel('Distance to Center of Halo, dr, (Kpc/h)')
plt.ylabel('Number of Galaxies')
plt.title('Number of Galaxies as a Function of Halo Radius')

GalH=np.array(([patch.get_height() for patch in patches]))
x1 = np.linspace(0,4000,50)
#print(GalH)
fit=np.polyfit(x1,GalH,2)
y=((fit[0]*(x1**2))+(fit[1]*x1)+fit[2])
plt.plot(x1,y)
print(fit)
