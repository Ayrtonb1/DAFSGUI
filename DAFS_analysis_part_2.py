# a sample .py file used in the context of a ZnFe2O4 tetrahedral A site sample, at the Zn K edge. 
# This is the 2nd part of a 2-step process. Please read the readme file accompanying this program for further guidance...

import numpy as np
import matplotlib.pyplot as plt
from hdf5_lib import *
import math
from larch.xafs import diffkk 
from scipy.integrate import quad_vec
from lmfit import Model
from hdf5_lib import *
import xraydb 

e = 2.718

def gaus_bkg(x,a,x0,sigma,bkg): #initial attempts to fit a Gaussian to each peak
    out = a*np.exp(-(x-x0)**2/(2*sigma**2)) + bkg
    return out

def bkg_func(x,a,b):
    model = a*x**2 + b
    return model

def two_d(two_theta,energy): #this is a function that calculates the two d values based on an input two theta array + energy single value, which returns the array two d values in Ang. may be worth improving the 12398 constant values accuracy
    out = (12398./energy)/np.sin(np.radians(two_theta/2))
    return out

def theta(two_d,energy):
    out = math.asin(12938. / (two_d * energy))
    return out

def norm_I(intensity,I_0): # this function will normalise the intensity by multiplying a single value I0 by an array of XRD intensity values
    out = (intensity/I_0)
    return out

def abs_corr(abs_tot,theta):
    out = (1-e**(-2*abs_tot))/(abs_tot * np.sin(theta)) #function to calculate D
    return out

def lorentz_corr(theta):
    out = 2*(np.sin(theta)**2)*np.cos(theta) #function to calculate L, then must multiply the observed intensity by L/D
    return out

#%% Step 1: importing 'raw DAFS' spectrum from part 1 of DAFS_analysis
filename = 'ZnFe2O4_400_results'

y_dat = []
ene  = []
with open(filename, "r") as mythen_file: #opening the XAS file called mainfile and appending the energies and corresponding file names, as well as the I_0 as 3 lists
    for line in mythen_file:
        if line[0] != "#":
                split_line = line.split()
                ene.append(split_line[0])
                y_dat.append(split_line[3])
y_new_shift = np.array(y_dat).astype(float)
ene = np.array(ene).astype(float)

y_new_shift = np.convolve(y_new_shift, np.ones(20)/20, mode = 'same') # Convolution, the box-car of which can be modified by changing the numbers in np.ones (as long as they match each other)
plt.plot(ene,y_new_shift, color = 'black', label = 'raw DAFS')
plt.legend()
plt.show()
#%% Step 2: importing accompanying XAS
XAS_filename = '624848_b18.dat'
ener = []
file_names = [] #this will be a list with n_energypoints pilatus file names
I_0 = []
exafs = []
with open(XAS_filename, "r") as mythen_file: #opening the XAS file and appending the energies and corresponding file names, as well as the I_0 as 3 lists
    for line in mythen_file:
        if line[0] != "#":
                split_line = line.split()  # list of space-separated substrings
                ener.append(float(split_line[0]))
                file_names.append(split_line[-1])
                I_0.append(split_line[2])
                exafs.append(split_line[5]) #These numbers can be modified as necessary depending on the nature of the user's XAS output file         
ene_2 = np.array(ener).astype(float)
I_0 = np.array(I_0).astype(float)
exafs = np.array(exafs).astype(float)
exafs = exafs[0:-1]
I_0 = I_0[0:-1]

# Only the first 900 datapoints taken due to a glitch. This can be modified by the user or even removed completely.
y_new_shift = y_new_shift[10:890]
ene = ene[10:890]
energies = ene[10:890]
I_0 = I_0[10:890]
exafs = exafs[10:890]

# Plotting the XAS
plt.plot(ene,exafs,color = 'black', label =  'exafs')
plt.xlabel("Energy / eV")
plt.ylabel("μ(E)")
plt.legend()
plt.show()

edge = 9658.53 # the user specified edge energy
edge = float(edge)
dhkl = 1.351 # the d-spacing value (only used for file-saving purposes)
dhkl = float(dhkl)
z_abs = int(30) # the user specified z_absorber integer

#%% Step 3: beginning the model, defining the model fit function 'intensity'

# calculating exafs inflexion point energy
deriv_exafs = np.gradient(exafs,ene)
max_exafs_deriv = max(deriv_exafs)
index_exafs_deriv = np.where(max_exafs_deriv <= deriv_exafs)
deriv_exafs_select = ene[index_exafs_deriv[0]]
print('exafs inflexion: ', deriv_exafs_select)

# Calculating sin_theta for model fitting below
sin_theta = []
for i in range(len(ene)):
    sin_theta.append( (12938. / (2* dhkl * ene[i])))
    # sin_theta = sin_theta * (180/np.pi)
sin_theta = np.asarray(sin_theta)
sin_theta = sin_theta 

y_new_shift = np.interp(ene+10.5,ene,y_new_shift) # Optional interpolation to align raw dafs with f' ,as explained in thesis 
exafs = np.interp(ene+10.5,ene, exafs) 

dkk=diffkk(ene, exafs, z= z_abs, edge='K', mback_kws={'e0':edge, 'order': 1})
dkk.kk() #doing diffkk to get better guesses as startijng values for f1/f2. 

f1 = xraydb.f1_chantler(30,ene)
f2 = xraydb.f2_chantler(30,ene) #chantler bare-atom energies for f1/f2 

#%% Model starts, to calculate fit to y_new_shift 
def intensity(en, phi, beta, t, exafs, sin_theta, fprime, fsec, scale=1, slope=0, offset=0):
    costerm = (np.cos(phi) + (beta*fprime))
    sinterm = (np.sin(phi) + (beta*fsec))
    return scale * (costerm**2 + sinterm**2) * ((1 - e**((-2*exafs*t)/sin_theta)) / (2*exafs)) + offset

imodel = Model(intensity, independent_vars=['en', 'fprime', 'fsec', 'exafs', 'sin_theta'])
params = imodel.make_params(scale=0.1, offset=0.5, slope=0, beta= 0.2, phi= 4, t = 1)

# optionally constrain parameters and force scale to be positive. we can constrain parameters as desired. E.G.:
# params['scale'].vary = False
# params['t'].vary = False
# params['scale'].max = 5
# params['scale'].min = 0
# params['slope'].vary = False
# params['beta'].min = 0
# params['offset'].vary = False
# params['phi'].max = 5
# params['phi'].min = 0

init_value = imodel.eval(params, en=ene, fprime=f1, fsec=f2, exafs = exafs, sin_theta = sin_theta)
result = imodel.fit(y_new_shift, params, en=ene, fprime=f1, fsec=f2, exafs = exafs, sin_theta = sin_theta)
print(result.fit_report())

phi = result.params.get('phi').value
beta = result.params.get('beta').value
I0 = result.params.get('scale').value
Ioff = result.params.get('offset').value
slope = result.params.get('slope')
t = result.params.get('t')

# Save parameters in an exported table
def namestr(obj, namespace):
    names = [name for name in namespace if namespace[name] is obj]
    name = [n for n in names if 'a' in n]
    return name[0]

with open(str(filename)+ "first_guess_table_output.txt","w") as setupfile:
    x = [phi,beta,I0,Ioff,t]
    for name in x:
        setupfile.write(namestr(name, globals()) + "=" + repr(name) +"\n")
    setupfile.close()

# plot the initial, first guess fits and the experimental data
plt.plot(ene[10:-10], (result.best_fit[10:-10]), '--', label='Model Fit', color = 'red')
plt.plot(ene[10:-10], (init_value[10:-10]), '--', label='init value (lmfit)', color = 'blue')
plt.plot(ene[10:-10], y_new_shift[10:-10], label='Raw DAFS Spectrum', color = 'black')

plt.xlabel("Energy / eV")
plt.ylabel("Intensity")
plt.legend(loc = "upper right")
plt.xticks(np.arange(9650,10300, step=50))
plt.xlim(9610,9870)
plt.savefig('ZnFe2O4_400_first_guess_model_fit.pdf')
plt.show()

# Save the above model info as a text file
np.savetxt(str(filename) + '_' + str(dhkl) + '_model_1st_guess_and_new_abscorr.txt',np.column_stack((ene,y_new_shift,result.best_fit)))
#%% Step 4: calculating f' and f" based on above fit

def f1_guess(fsec, I, I0, phi, beta, Ioff,t, exafs, sin_theta):
    f1_guess = (1/beta) * (-(np.sqrt(((I-Ioff)/(I0*((1 - e**((-2*exafs*t)/sin_theta)) / (2*exafs)))) - (math.sin(phi) + beta*fsec)**2))  - math.cos(phi))
    return f1_guess
 
f1_minus = f1_guess(f2, y_new_shift, I0, phi, beta, Ioff, t,exafs,sin_theta)

plt.plot(ene,f1_minus)
plt.plot(ene,f1)
plt.show()

def f2_guess(fprime, I, I0, phi, beta, Ioff,t, exafs, sin_theta):
    f2_guess =  (1/beta)* (-(np.sqrt(((((I)-Ioff))/(I0*((1 - e**((-2*exafs*t)/sin_theta)) / (2*exafs)))) -((math.cos(phi)) +(beta*fprime))**2)  + (math.sin(phi))))
    return f2_guess

f2_minus = f2_guess(f1, y_new_shift, I0, phi, beta, Ioff, t,exafs,sin_theta)

plt.plot(ene,f2_minus,label = 'f2')
plt.legend()
plt.show()

import pandas as pd
df =pd.DataFrame(f1_minus)
df.fillna(method='ffill', inplace=True) # remove any zero values that may arise due to noise
#%% Calculating KK transforms of the above f' and f". Note that one pair will be preferred over the other
ene = np.array(ene)
ene_dash = np.array(ene)

i = (len(y_new_shift) - 1) 

# KK transformed f'
f1KK = lambda ene_dash : (f2_minus - f2) / ((ene_dash-1)**2 - (ene[i])**2)
f1KK_arr = []
f1KK_arr, err = (quad_vec(f1KK,(ene_dash[0]),(ene_dash[-1])))
f1_KK = (f1 - ((2*ene/math.pi) * f1KK_arr))

plt.plot(ene,f1_KK, label = 'f1_KK')
plt.plot(ene,f1, label = 'molecular f1')
plt.plot(ene,f1, label = 'atomic f1')
plt.legend()
plt.show()

# KK transformed f"
f2KK = lambda ene_dash : (f1_minus - f1) / ((ene_dash-1)**2 - (ene[i])**2)
f2KK_arr = []
f2KK_arr, err = (quad_vec(f2KK,(ene_dash[0]),(ene_dash[-1])))
fsecond_KK = (f2 - ((2*ene/math.pi) * f2KK_arr))

plt.plot(ene,fsecond_KK, label = 'new f2')
plt.plot(ene,f2, label = 'atomic f2')
plt.plot(ene,f1_minus, label = 'new f1')
plt.plot(ene,f1, label = 'atomic f1 HERE?')
plt.legend()
plt.show()

# optional kk transform of f" to generate an f' 
dkk=diffkk(ene, fsecond_KK, z= z_abs, edge='K', mback_kws={'e0':edge, 'order': 1})
dkk.kk()

plt.plot(ene,dkk.fp, label = 'f2_minus dkk fp')
plt.plot(ene,dkk.f1)
plt.legend()
plt.show()
plt.plot(ene,dkk.fpp, label = 'f2_minus dkk fpp')
plt.plot(ene,dkk.f2)
plt.legend()
plt.show()

# saving the above f" - names can be changed in this command as desired
np.savetxt(str(filename) + '_' + str(dhkl) + '_fsec',np.column_stack((ene,fsecond_KK, f2, f2)))

# A more comprehensive plot of the above. Also saves as figure if desired
plt.plot(ene,f1_minus, color = 'black')
plt.plot(ene,fsecond_KK, color = 'black')
plt.plot(ene,f1, '--', color = 'black')
plt.plot(ene,f2, '--', color = 'black')
plt.ylabel('Intensity')
plt.xlabel('Energy / eV')
plt.xticks(np.arange(9650,10300, step=50))
plt.xlim(9610,9870)
plt.yticks(np.arange(-10,9, step=5))
plt.savefig(filename + 'ZnFe2o4_400_fsec_fprime_first_guess.pdf')
# plt.legend()
plt.show()

#%% step 5: iteration of the above process from step 3 and 4. The desired f'/f" pairs can be readily exchanged as the user desires based on the above calculated results for f'/f".
imodel = Model(intensity, independent_vars=['en', 'fprime', 'fsec', 'exafs', 'sin_theta'])
params = imodel.make_params(scale=I0, offset=Ioff, slope=0, beta=beta, phi= phi, t = t)
params['scale'].max = 5
params['scale'].min = 0
params['t'].vary = False

result = imodel.fit(y_new_shift, params, en=ene, fprime=f1_minus, fsec=fsecond_KK, exafs = exafs, sin_theta = sin_theta)
print(result.fit_report())

phi = result.params.get('phi').value 
beta = result.params.get('beta').value
I0 = result.params.get('scale').value
Ioff = result.params.get('offset').value
slope = result.params.get('slope')
print(' I0 = ',I0, '\n','phi = ', phi, '\n','beta =',beta, '\n','Ioff =', Ioff, '\n','energy dependence =', slope,'\n','t =', t)

def namestr(obj, namespace):
    names = [name for name in namespace if namespace[name] is obj]
    name = [n for n in names if 'a' in n]
    return name[0]

with open(str(filename)+ "table_output.txt","w") as setupfile:
    x = [phi,beta,I0,Ioff,t]
    for name in x:
        setupfile.write(namestr(name, globals()) + "=" + repr(name) +"\n")
    setupfile.close()

plt.plot(ene[10:-10], (result.best_fit[10:-10]), '--', label='Model Fit', color = 'black')
plt.plot(ene[10:-10], y_new_shift[10:-10], label='Raw DAFS Spectrum', color = 'black')

plt.xlabel("Energy / eV")
plt.ylabel("Intensity")
plt.legend(loc = "lower right")
plt.xticks(np.arange(9650,10300, step=50))
plt.xlim(9610,9870)
plt.savefig('ZnFe2O4_400_final_guess_model_fit.pdf')
plt.show()

f1_minus = f1_guess(fsecond_KK, y_new_shift, I0, phi, beta, Ioff, t,exafs,sin_theta)
i = (len(y_new_shift) - 1) 
ene = np.array(ene)
ene_dash = np.array(ene)

f2KK = lambda ene_dash : ((f1_minus - f1)/ ((ene_dash-1)**2 - (ene[i])**2))
f2KK_arr = []
f2KK_arr, err = (quad_vec(f2KK,(ene_dash[0]),(ene_dash[-1])))
fsecond_KK = (f2- ((2*ene/math.pi) * f2KK_arr)) 

# plotting:
plt.plot(ene,fsecond_KK, color = 'black')
plt.ylabel('Intensity')
plt.xlabel('Energy / eV')
plt.xticks(np.arange(9500,9850, step=50))
plt.xlim(min(ene),9850,100)
plt.yticks(np.arange(-2,7, step=2))
plt.ylim(-1,7)
plt.savefig(filename + 'ZnFe2o4_400_fsec_final_final.pdf')
plt.show()

plt.plot(ene,fsecond_KK, color = 'black')
plt.plot(ene,f1_minus, color = 'black')
plt.ylabel('Intensity')
plt.xlabel('Energy / eV')
plt.xticks(np.arange(9500,10000, step=50))
plt.xlim(min(ene),9850,200)
plt.yticks(np.arange(-10,7, step=5))
plt.ylim(-10,7)
plt.savefig(filename + 'ZnFe2O4_400_fsec_fprime_final.pdf')
plt.show()

# # 2 other alternative f'/f" calculations
# # finding the pair of f" using diffkk.
# f2_new = f2_guess(f1_KK, y_new_shift, I0, phi, beta, Ioff, t,exafs,sin_theta)
# i = (len(y_new_shift) - 1) 
# f2_minus = f2_new
# dkk_f2 = f2
# dkk=diffkk(ene, f2_minus, z= z_abs, edge='K', mback_kws={'e0':edge, 'order': 1})
# dkk.kk()

# plt.plot(ene,dkk.fp, label = 'f2_minus dkk fp')
# plt.plot(ene,dkk.f1)
# plt.legend()
# plt.show()
# plt.plot(ene,dkk.fpp, label = 'f2_minus dkk fpp')
# plt.plot(ene,dkk.f2)
# plt.legend()
# plt.show()

# # optional f1 transform. f2 is transformed here so this has been commented out.
# f1KK = lambda ene_dash : ((f2_minus - dkk.fpp)/ ((ene_dash-1)**2 - (ene[i])**2))
# f1KK_arr = []
# f1KK_arr, err = (quad_vec(f1KK,(ene_dash[0]),(ene_dash[-1])))
# f1_KK = (dkk.fp - ((2*ene/math.pi) * f1KK_arr)) 

# # plotting:
# plt.plot(ene,f1_KK, label = 'new f1 updated')
# plt.legend()
# plt.plot(ene,f1, label = 'atomic f1')
# plt.show()
# plt.plot(ene,f2, label = 'atomic f2')
# plt.plot(ene,f2_minus, label = 'new f2 updated')
# plt.legend()
# plt.show()

