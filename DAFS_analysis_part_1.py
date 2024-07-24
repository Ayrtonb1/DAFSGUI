
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from hdf5_lib import *
import sys
import math
from lmfit.models import PseudoVoigtModel, LinearModel
from scipy.integrate import trapz

def gaus_bkg(x,a,x0,sigma,bkg): #initial attempts to fit a Gaussian to each peak
    out = a*np.exp(-(x-x0)**2/(2*sigma**2)) + bkg
    return out

e = 2.718

def bkg_func(x,a,b):
    model = a*x**2 + b
    return model

def resid(p0):
    return I - bkg_func(x2,p0[0],p0[1])

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

def intensity(en, phi, beta, t, exafs, sin_theta, scale=1, slope=0, offset=0, fprime=-1, fsec=1):
    costerm = (np.cos((phi)) + (beta*fprime))
    sinterm = (np.sin((phi)) + (beta*fsec))
    return scale * ((costerm**2 + sinterm**2)) * (1 - e**((-2*exafs*t)/sin_theta)) / (2*exafs) + offset

def findCircle(x1, y1, x2, y2, x3, y3) :

                x12 = x1 - x2
                x13 = x1 - x3
                y12 = y1 - y2
                y13 = y1 - y3
                y31 = y3 - y1
                y21 = y2 - y1
                x31 = x3 - x1
                x21 = x2 - x1

                sx13 = pow(x1, 2) - pow(x3, 2);
                sy13 = pow(y1, 2) - pow(y3, 2);
                sx21 = pow(x2, 2) - pow(x1, 2);
                sy21 = pow(y2, 2) - pow(y1, 2);

                f = (((sx13) * (x12) + (sy13) *
                              (x12) + (sx21) * (x13) +
                              (sy21) * (x13)) // (2 *
                              ((y31) * (x12) - (y21) * (x13))));    
                g = (((sx13) * (y12) + (sy13) * (y12) +
                              (sx21) * (y13) + (sy21) * (y13)) //
                              (2 * ((x31) * (y12) - (x21) * (y13))));
                c = (-pow(x1, 2) - pow(y1, 2) -
                              2 * g * x1 - 2 * f * y1);
               
                # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
                # where centre is (h = -g, k = -f) and
                # radius r as r^2 = h^2 + k^2 - c
                h = -g;
                k = -f;
                sqr_of_r = h * h + k * k - c;

                # r is the radius
                r = round(sqrt(sqr_of_r), 5);
                print("Centre = (", h, ", ", k, ")");
                print("Radius = ", r);
                return h,k,r 

def radial_profile(data, center):
    # probably faster implementation here https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    # r = int(r)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr;
    return radialprofile

def find_data_in_ROI(x,y,z, Cen_x,Cen_y):
    coeff_line_up = np.polyfit([Cen_x, max(x)],[Cen_y, max(y)],1);
    coeff_line_down=np.polyfit([Cen_x, max(x)],[Cen_y, min(y)],1);
    line_up=np.polyval(coeff_line_up,x); line_down=np.polyval(coeff_line_down,x);
    reduced_data = z
    for index1 in range(len(x)):
        for index2 in range(len(y)):
            if y[index2] > line_up[index1] :
                reduced_data[index2,index1]=0;
            if y[index2] < line_down[index1] :
                reduced_data[index2,index1]=0;
    return reduced_data

def shift(xs, n):
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = 0
        e[n:] = xs[:-n]
    else:
        e[n:] = 0
        e[:n] = xs[-n:]
    return e

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def f1_guess(fsec, I, I0, phi, beta, Ioff, abscorr):
    f1_guess = (1/beta) * +(np.sqrt(((((I)-Ioff))/(I0*abscorr)) +((math.sin(phi)) +(beta*fsec))**2)  - (math.cos(phi)))
    return f1_guess

#%% Step 1: importing the pilatus detector data

filename = '624848_ZnFe2O4_kapton4layers.hdf5'

hdf_print_names(filename)
"""""
entry
entry/data
entry/data/data
entry/instrument
entry/instrument/NDAttributes
entry/instrument/NDAttributes/AcqPeriod
entry/instrument/NDAttributes/CameraManufacturer
entry/instrument/NDAttributes/CameraModel
entry/instrument/NDAttributes/DriverFileName
entry/instrument/NDAttributes/ImageCounter
entry/instrument/NDAttributes/ImageMode
entry/instrument/NDAttributes/MaxSizeX
entry/instrument/NDAttributes/MaxSizeY
entry/instrument/NDAttributes/NDArrayEpicsTSSec
entry/instrument/NDAttributes/NDArrayEpicsTSnSec
entry/instrument/NDAttributes/NDArrayTimeStamp
entry/instrument/NDAttributes/NDArrayUniqueId
entry/instrument/NDAttributes/NumExposures
entry/instrument/NDAttributes/NumImages
entry/instrument/NDAttributes/TIFFImageDescription
entry/instrument/NDAttributes/TriggerMode
entry/instrument/detector
entry/instrument/detector/NDAttributes
entry/instrument/performance
entry/instrument/performance/timestamp
"""
#XRD is a 3D array containing an 'n_energypoints' number of images, each image being an XRD at 1 energy
XRD=hdf_get_item(filename,'entry/data/data')
z = XRD[0,:,:]

image=hdf_get_item(filename,'entry/data/data')

n_energypoints= shape(XRD)[0]
n_column = shape(XRD) [1]
n_row = shape(XRD) [2]

# printing all corresponding maximum intensities for each column
for i in range(n_column):
    buf=z[i,:]; index1=np.where(buf==max(buf));
    print(i, index1)

y=np.linspace(0,z.shape[0],(z.shape[0]));
x=np.linspace(0,z.shape[1],(z.shape[1]));
 
# selecting 3 points of a circle based on the results from the maximum intensities for each column calculation above
buf1 = 36
buf2 = 103
buf3 = 167

buf=z[buf1,:]; index1=np.where(buf==max(buf)); #defining indices for the circle from buf1, buf2, buf3
buf=z[buf2,:]; index2=np.where(buf==max(buf));
buf=z[buf3,:]; index3=np.where(buf==max(buf));

print('number of energy points: ', n_energypoints)
print(index1,index2,index3)

if (np.abs(index2[0])-index1[0] + np.abs(index2[0]-index3[0])) < 15: #10 pixels is an arbitrary width   
    Cen_x, Cen_y, rad = findCircle(int(index1[0]),buf1, int(index2[0]),buf2, int(index3[0]),buf3)
else:
    print("automatic estimation failed")

#%% Step 2: Extracting the XRD patterns iteratively from each pilatus image
extracted_xrd = np.ones([len(x),np.shape(XRD)[0]])

print(n_energypoints)
for index in range(n_energypoints):
    z=XRD[index,:,:]
    y=np.linspace(0,z.shape[0],(z.shape[0]));
    x=np.linspace(0,z.shape[1],(z.shape[1]));
    reduced_data = find_data_in_ROI(x,y,z, Cen_x,Cen_y)
    buffer = radial_profile(reduced_data, [Cen_x,Cen_y]) 
    buffer[np.isnan(buffer)] = 0
    buffer=buffer[np.abs(Cen_x-1):-1]; 
    buffer = buffer[0:1475]
    extracted_xrd[:,index]=buffer
    print('iteration', index)

#%% Step 3: Importing accompanying XAS
filename = '624848_b18.dat'

energies = []
file_names = [] #this will be a list with size n_energypoints filled with pilatus file names
I_0 = []
exafs = []
with open(filename, "r") as mythen_file: #opening the XAS file and appending the energies and corresponding file names, as well as the I_0 as 3 lists
    for line in mythen_file:
        if line[0] != "#":
                split_line = line.split()  # list of space-separated substrings
                energies.append(float(split_line[0]))
                file_names.append(split_line[-1])
                I_0.append(split_line[2])
                exafs.append(split_line[5])
            
ene = np.array(energies).astype(float)
I_0 = np.array(I_0).astype(float)
exafs = np.array(exafs).astype(float)

plt.plot(ene, exafs, color = 'black')
plt.xlabel("Energy / eV")
plt.ylabel("μ(E)")
plt.savefig(filename + 'exafs.pdf')
plt.show()

#%% Step 4: 2theta calibration and plotting XRD

#An elementary calibration based on the x and y values of 4 peaks, to convert the x-axis to 2theta. May not always be 100% accurate but will be close and will allow the user to visualise the full spectrum and specify their own window for thte subsequent peak intensity extraction
x1 = [214, 337, 930, 808]
y = [24.988, 29.383, 51.266, 46.836]

coeff = np.polyfit(x1,y,1)
x2 = np.polyval(coeff,x)

# converting to d and plotting all XRD in terms of d
for i in range(n_energypoints):
            d = (two_d(x2, ene[i])) / 2
for i in y:
    alignment_factor = 0 #optional alignment factor to move the XRD to be in the desired d-range
    plt.plot(d+alignment_factor, extracted_xrd)
    plt.xlabel("d spacing / Å")
    plt.ylabel("Diffracted Peak Intensity")
plt.savefig(filename + 'XRD_d.png')
plt.show()

# plotting all XRD in 2theta
for i in y:
    plt.plot(x2, extracted_xrd)
    plt.xlabel("Angle / 2θ")
    plt.ylabel("Diffracted Peak Intensity")
plt.savefig(filename + 'XRD_2theta.pdf')
plt.show()
#%% step 5: extracting intensities from a given peak
dhkl = 2.9969 # the dhkl value of the reflection we are concerned with
dhkl = [dhkl]

for dhkl in dhkl: # take a given d value as specified by system argument, and use it to obtain a series of 'max_y' values within a given angle range
    alignment_factor = 0
    area_arr  = []
    a = []
    for i in range(n_energypoints):
            print(i)
            I = extracted_xrd[:,i]
            x = x2
            dhkl = float(dhkl)
            d = ((two_d(x, ene[i])) / 2) + alignment_factor # converts two theta to d using the two_d function
            
            dmin = float(1.8)
            dmax = float(1.9) #user should define these based on their desired window 
            index_d = np.where((d >= dmin) & (d <= dmax))  # stores the number of values of d, that is, the index of d, between the specified range, i.e. the 12th, 13th, 14th value
            d_select = d[index_d[0]] 
            y_select = I[index_d[0]]# must also specify this for y, so lists are same length
            y_select = y_select - min(y_select)

            if i == 0: #defining parameters for the first fit only 
                n = len(d_select) 
                mean = sum(d_select) / n
                bkg = min(j for j in y_select if j > 0) #A nested loop to only take bkg values above 0 to avoid accounting for any glitches 
                sigma = np.sqrt(sum((y_select-bkg)*(d_select-mean)**2) / sum(y_select))
                amp = max(y_select)
                area_guess = 0.5
                
            elif i != 0: #defining parameters for the rest of the fits so that they start with the parameters before
                amp = out.params.get('pv_height').value
                area_guess = out.params.get('pv_amplitude').value
                mean = out.params.get('pv_center').value
                sigma = out.params.get('pv_sigma').value

            mod = PseudoVoigtModel(prefix = 'pv_') # pseudo-voigt peak component
            params = mod.make_params(pv_amplitude = area_guess, pv_height = amp, pv_center=mean, pv_sigma=sigma)

            mod_lin = LinearModel(prefix= 'lin_') # linear component
            params.update( mod_lin.guess(y_select, x=d_select))

            # sum components to make a composite model (add more if needed)
            model  =  mod + mod_lin
            
            init = model.eval(params, x=d_select)
            out = model.fit(y_select, params, x=d_select)
            
            print(out.fit_report())
            comps = out.eval_components(x=d_select)
                    
            res = (out.best_fit) 
            area = (trapz(comps['pv_']))  #trapezoidal integration of the pv component
            a.append(out.params.get('pv_amplitude').value)
            area_arr.append(area)
            
            if i in np.arange(1,n_energypoints, step=20): #plots d in the specified window, with the data subtraction, for every 20th dataset
                plt.plot(d_select, y_select, color = 'black', label = 'Experimental XRD')
                plt.plot(d_select, comps['pv_'], color = 'red',label='Pseudo-Voigt Fit')
                plt.plot(d_select, comps['lin_'], '--',color = 'black', label='Background')
                plt.plot(d_select,res, '--', color = 'green', label = 'total fit')
                plt.xlim(min(d_select),max(d_select))

                plt.legend(loc = "upper left")
                plt.xlabel("d-spacing / Å")
                plt.ylabel("Diffracted peak intensity")
                plt.legend()
                plt.savefig(filename + 'XRD_bkg_sub.pdf')
                plt.show()

    area_arr = np.array(area_arr)
    area_norm = norm_I(area_arr,I_0)
    a_norm = norm_I(a,I_0)
    y_new = abs((area_norm)) / max(area_norm)
    a_new = abs((a_norm))

    mask = np.isnan(y_new)
    idx = np.where(~mask,np.arange(mask.shape[0]),0)
    np.maximum.accumulate(idx, out=idx)
    y_new = y_new[idx] #gets rid of any values less than 0
    
    edge = float(9658.53) # user specified edge
    y_new_shift = y_new
    
    # plotting and saving final result
    plt.plot(ene, y_new_shift,  color = 'black')
    plt.xlabel("Energy / eV")
    plt.ylabel("Diffracted Intensity")
    plt.savefig(filename + 'raw_DAFS.pdf')
    plt.show()
    
    np.savetxt(str(filename) + '_' + str(dhkl) + '_results',np.column_stack((ene,I_0,area_norm,y_new)))
