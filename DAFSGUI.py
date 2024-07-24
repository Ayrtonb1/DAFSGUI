# Guidance -
# Things to change for each dataset:
    #change filename (1)
    #change configurations per file/sys.argvs (2)
    #change angle correction and dr range (3)
    #change / interpolate I0 array length (4)
    #Change f1/f2 edges (5)
    #Check that bkg function is accurate. If not play around with the points 
    
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from hdf5_lib import *
import sys
import math
from larch.xafs import diffkk 
from scipy.integrate import quad_vec
from lmfit.models import PseudoVoigtModel, LinearModel
from lmfit import Model
from scipy.integrate import trapz
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtWidgets
# from PyQt5.QtCore import QFile, QFileInfo, QRect

def gaus_bkg(x,a,x0,sigma,bkg): #initial attempts to fit a Gaussian to each peak
    out = a*np.exp(-(x-x0)**2/(2*sigma**2)) + bkg
    return out

e = 2.718

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

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class Ui_MainWindow(QtWidgets.QMainWindow): # as opposed to default object
    def setupUi(self, MainWindow):
        
        MainWindow.setObjectName("DAFS GUI")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(85, 5, 60, 16))
        self.title.setObjectName("self.title")
        self.title.setStyleSheet("font-weight: bold")
        
        self.XRD_button_1 = QtWidgets.QPushButton(self.centralwidget)
        self.XRD_button_1.setGeometry(QtCore.QRect(80, 110, 113, 32))
        self.XRD_button_1.setObjectName("XRD_button_1")
        self.XRD_button_2 = QtWidgets.QPushButton(self.centralwidget)
        self.XRD_button_2.setGeometry(QtCore.QRect(200, 110, 113, 32))
        self.XRD_button_2.setObjectName("XRD_button_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(90, 90, 60, 16))
        self.label.setObjectName("label")
        self.XRD_button_3 = QtWidgets.QPushButton(self.centralwidget)
        self.XRD_button_3.setGeometry(QtCore.QRect(80, 150, 113, 32))
        self.XRD_button_3.setObjectName("XRD_button_3")
        
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(320, 90, 60, 16))
        self.label_2.setObjectName("label_2")
        self.EXAFS_button_1 = QtWidgets.QPushButton(self.centralwidget)
        self.EXAFS_button_1.setGeometry(QtCore.QRect(320, 110, 113, 32))
        self.EXAFS_button_1.setObjectName("EXAFS_button_1")
        self.EXAFS_button_2 = QtWidgets.QPushButton(self.centralwidget)
        self.EXAFS_button_2.setGeometry(QtCore.QRect(440, 110, 113, 32))
        self.EXAFS_button_2.setObjectName("EXAFS_button_2")
        
        self.abscorr_button = QtWidgets.QPushButton(self.centralwidget)
        self.abscorr_button.setGeometry(QtCore.QRect(320, 150, 113, 32))
        self.abscorr_button.setObjectName("pushButton_6")
        
        self.window_label = QtWidgets.QLabel(self.centralwidget)
        self.window_label.setGeometry(QtCore.QRect(90, 190, 50, 10))
        self.window_label.setObjectName("window_label")
        
        self.window_text = QtWidgets.QLineEdit(self.centralwidget)
        self.window_text.setGeometry(QtCore.QRect(90, 210, 111, 31))
        self.window_text.setObjectName("window_text")
        
        self.smooth_label = QtWidgets.QLabel(self.centralwidget)
        self.smooth_label.setGeometry(QtCore.QRect(320, 255, 60, 16))
        self.smooth_label.setObjectName("smooth_label")
        
        self.smooth_button_1 = QtWidgets.QPushButton(self.centralwidget)
        self.smooth_button_1.setGeometry(QtCore.QRect(320, 280, 140, 32))
        self.smooth_button_1.setObjectName("smooth_button_1")
        
        self.smooth_button_2 = QtWidgets.QPushButton(self.centralwidget)
        self.smooth_button_2.setGeometry(QtCore.QRect(320, 320, 140, 32))
        self.smooth_button_2.setObjectName("smooth_button_2")
        
        self.dminmax_label = QtWidgets.QLabel(self.centralwidget)
        self.dminmax_label.setGeometry(QtCore.QRect(320, 190, 60, 16))
        self.dminmax_label.setObjectName("dminmax_label")
        
        self.d_min = QtWidgets.QLineEdit(self.centralwidget)
        self.d_min.setGeometry(QtCore.QRect(353, 210, 101, 31))
        self.d_min.setObjectName("d_min")
        
        self.d_max = QtWidgets.QLineEdit(self.centralwidget)
        self.d_max.setGeometry(QtCore.QRect(475, 210, 101, 31))
        self.d_max.setObjectName("d_max")
        
        self.d_label = QtWidgets.QLineEdit(self.centralwidget)
        self.d_label.setGeometry(QtCore.QRect(90, 50, 111, 31))
        self.d_label.setObjectName("d_label")

        self.z_label = QtWidgets.QLineEdit(self.centralwidget)
        self.z_label.setGeometry(QtCore.QRect(210, 50, 111, 31))
        self.z_label.setObjectName("z_label")
        
        self.E_label = QtWidgets.QLineEdit(self.centralwidget)
        self.E_label.setGeometry(QtCore.QRect(330, 50, 101, 31))
        self.E_label.setObjectName("E_label")
        
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(330, 30, 60, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(210, 30, 60, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(90, 30, 60, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(90, 255, 60, 16))
        self.label_6.setObjectName("label_6")
        
        self.DAFS_button_2_1 = QtWidgets.QPushButton(self.centralwidget)
        self.DAFS_button_2_1.setGeometry(QtCore.QRect(80, 320, 113, 32))
        self.DAFS_button_2_1.setObjectName("DAFS_button_1")
        self.DAFS_button_2_2 = QtWidgets.QPushButton(self.centralwidget)
        self.DAFS_button_2_2.setGeometry(QtCore.QRect(200, 320, 113, 32))
        self.DAFS_button_2_2.setObjectName("DAFS_button_2")
        self.DAFS_button_1 = QtWidgets.QPushButton(self.centralwidget)
        self.DAFS_button_1.setGeometry(QtCore.QRect(80, 280, 113, 32))
        self.DAFS_button_1.setObjectName("DAFS_button_2_1")
        self.DAFS_button_2 = QtWidgets.QPushButton(self.centralwidget)
        self.DAFS_button_2.setGeometry(QtCore.QRect(200, 280, 113, 32))
        self.DAFS_button_2.setObjectName("DAFS_button_2_2")
        self.DAFS_button_3 = QtWidgets.QPushButton(self.centralwidget)
        self.DAFS_button_3.setGeometry(QtCore.QRect(80, 360, 113, 32))
        self.DAFS_button_3.setObjectName("DAFS_button_3")
        self.save_btn = QtWidgets.QPushButton(self.centralwidget)
        self.save_btn.setGeometry(QtCore.QRect(200, 360, 113, 32))
        self.save_btn.setObjectName("save_btn")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        #lines and borders
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(80, 85, 500, 5))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(80, 185, 500, 5))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(80, 248, 500, 5))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(315, 85, 5, 331))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)

        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(80, 25, 500, 5))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(80, 415, 500, 5))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)

        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(578, 25, 5, 391))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(80, 25, 5, 391))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        
        self.title.setText(_translate("MainWindow", "DAFS Data Analysis GUI"))
        self.title.adjustSize() 
        
        self.XRD_button_1.setText(_translate("MainWindow", "Import"))
        self.XRD_button_2.setText(_translate("MainWindow", "Extract"))
        self.label.setText(_translate("MainWindow", "Experimental XRD File (hdf5,h5 etc.)"))
        self.label.adjustSize() 
        self.XRD_button_3.setText(_translate("MainWindow", "Plot XRD"))
        self.label_2.setText(_translate("MainWindow", "Experimental EXAFS File (.dat, .txt, etc.)"))
        self.label_2.adjustSize() 
        self.EXAFS_button_1.setText(_translate("MainWindow", "Import"))
        self.EXAFS_button_2.setText(_translate("MainWindow", "Plot EXAFS"))
        self.abscorr_button.setText(_translate("MainWindow", "Abs. Corr."))
        
        self.label_3.setText(_translate("MainWindow", "E<sub>edge </sub> / eV"))
        self.label_4.setText(_translate("MainWindow", "Z<sub>absorber"))
        self.label_5.setText(_translate("MainWindow", "Reflection d-spacing / Å"))
        self.label_6.setText(_translate("MainWindow", "DAFS - Extracting f\":"))
        
        self.window_label.setText(_translate("MainWindow", "Enter Shift Value In Ang. (optional):"))
        self.window_label.adjustSize() 
        
        self.smooth_label.setText(_translate("MainWindow", "Import smoothed DAFS data (optional)"))
        self.smooth_label.adjustSize() 
        
        self.smooth_button_1.setText(_translate("MainWindow", "Import"))
        self.smooth_button_2.setText(_translate("MainWindow", "Save Raw DAFS"))
        
        self.dminmax_label.setText(_translate("MainWindow", "Define d-space Window Range: \n\nfrom:                            to:"))
        self.dminmax_label.adjustSize() 
        
        self.label_6.adjustSize() 
        
        self.DAFS_button_1.setText(_translate("MainWindow", "Raw DAFS"))
        self.DAFS_button_2.setText(_translate("MainWindow", "Model"))
        self.DAFS_button_2_1.setText(_translate("MainWindow", "1st guess f\'"))
        self.DAFS_button_2_2.setText(_translate("MainWindow", "1st guess f\""))
        self.DAFS_button_3.setText(_translate("MainWindow", "Iterate"))

        self.save_btn.setText(_translate("MainWindow", "Save As.."))
        
        self.abscorr_button.clicked.connect(self.abscorr)
        
        self.XRD_button_1.clicked.connect(self.open_file)
        
        self.EXAFS_button_1.clicked.connect(self.exafs_file)
        
        self.EXAFS_button_2.clicked.connect(self.plot_exafs)
        
        self.XRD_button_2.clicked.connect(self.extract_xrd)
        
        self.DAFS_button_1.clicked.connect(self.get_peak_areas)
        
        self.XRD_button_3.clicked.connect(self.plot_xrd)
        
        self.DAFS_button_2.clicked.connect(self.first_guess_model)
        
        self.DAFS_button_2_1.clicked.connect(self.first_guess_f1)
        
        self.DAFS_button_2_2.clicked.connect(self.first_guess_f2)
        
        self.DAFS_button_3.clicked.connect(self.iterate)
        
        self.save_btn.clicked.connect(self.save)
        
        self.smooth_button_1.clicked.connect(self.import_smoothed)
        
        self.smooth_button_2.clicked.connect(self.save_raw)
        
    def import_smoothed(self):
        
        filename = QFileDialog.getOpenFileName(MainWindow, 'Open File')
        self.filename = filename[0]
        
        y_dat = []
        ene  = []
        with open(self.filename, "r") as mythen_file: #opening the XAS file called mainfile and appending the energies and corresponding file names, as well as the I_0 as 3 lists
            for line in mythen_file:
                if line[0] != "#":
                        split_line = line.split()
                        ene.append(split_line[0])
                        y_dat.append(split_line[3])

        y_dat = np.array(y_dat).astype(float)
        print('y_dat',y_dat)
        self.ene = np.array(ene).astype(float)
        self.n_energypoints = len(self.ene)
        
        y_dat = y_dat[0:self.n_energypoints]
        y_dat = -y_dat
        self.y_new_shift =  abs(y_dat / abs(max(y_dat)))
        
        min_y_new_shift = min(self.y_new_shift)
        index_y_new_shift = np.where(self.y_new_shift <= min_y_new_shift)
        y_new_shift_select = self.ene[index_y_new_shift[0]]
        print('y_new_shift ', y_new_shift_select)
        
        self.edge = self.E_label.text()
        self.edge = float(self.edge)
        self.difference = y_new_shift_select - self.edge
        
        plt.plot(self.ene, self.y_new_shift)
        plt.show()
        
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.plot(self.ene,self.y_new_shift, color = 'black')
        sc.axes.set_xlabel("Energy / eV")
        sc.axes.set_ylabel("Intensity")
        sc.axes.grid(True)
        self.setCentralWidget(sc)
        
        toolbar = NavigationToolbar(sc, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(sc)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.show()

        self.msg = QMessageBox()
        self.msg.setText("Done!")
        
        x = self.msg.exec_()

    def save_raw(self):
        
        DAFS_for_smoothing = 1/self.y_new_shift
        np.savetxt(str(self.dhkl) + 'DAFS_for_smoothing_ATHENA',np.column_stack((self.ene,DAFS_for_smoothing)))
        
        x = self.msg.exec_()
        
    def open_file(self): #%% Step 1: importing the pilatus detector data
        
        filename = QFileDialog.getOpenFileName(MainWindow, 'Open File')
        self.filename = filename[0]

        hdf_print_names(self.filename)
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
        # a 3D array containing an 'n_energypoints' number of images, each image being an XRD at 1 incremental energy

        self.XRD=hdf_get_item(self.filename,'entry/data/data')
        z = self.XRD[0,:,:]

        image=hdf_get_item(self.filename,'entry/data/data')

        self.n_energypoints= shape(self.XRD)[0]
        n_column = shape(self.XRD) [1]
        n_row = shape(self.XRD) [2]
        print(n_column,n_row)
        
        # printing all corresponding maximum intensities for each column
        for i in range(n_column):
            buf=z[i,:]; index1=np.where(buf==max(buf));
            print(i, index1)
        
        self.y=np.linspace(0,z.shape[0],(z.shape[0]));
        self.x=np.linspace(0,z.shape[1],(z.shape[1]));
        
        # selecting 3 points of a circle based on the results from the maximum intensities for each column calculation above
        self.buf1 = 37
        self.buf2 = 103
        self.buf3 = 155
        
        buf=z[self.buf1,:]; index1=np.where(buf==max(buf)); #defining indices for the circle from buf1, buf2, buf3
        buf=z[self.buf2,:]; index2=np.where(buf==max(buf));
        buf=z[self.buf3,:]; index3=np.where(buf==max(buf));

        print('number of energy points: ', self.n_energypoints)
        print(index1,index2,index3)

        if (np.abs(index2[0])-index1[0] + np.abs(index2[0]-index3[0])) < 15:
            self.Cen_x, self.Cen_y, rad = findCircle(int(index1[0]),self.buf1, int(index2[0]),self.buf2, int(index3[0]),self.buf3)
        else:
            self.msg2 = QMessageBox()
            self.msg2.setText('Automatic estimation of the centre failed')
            
            x = self.msg2.exec_()

        self.extracted_xrd=np.ones([len(self.x),np.shape(self.XRD)[0]])
        
        self.msg = QMessageBox()
        self.msg.setText("Done!")
        
        x = self.msg.exec_()
        
    def extract_xrd(self): # Step 2: Extracting the XRD patterns iteratively from each pilatus image
        
        for index in range(self.n_energypoints):
            z=self.XRD[index,:,:]
            self.y=np.linspace(0,z.shape[0],(z.shape[0]));
            self.x=np.linspace(0,z.shape[1],(z.shape[1]));
            reduced_data = find_data_in_ROI(self.x,self.y,z, self.Cen_x,self.Cen_y)
            buffer = radial_profile(reduced_data, [self.Cen_x,self.Cen_y]) 
            buffer[np.isnan(buffer)] = 0
            buffer=buffer[np.abs(self.Cen_x):-1]; 
            buffer = buffer[0:1475]
            self.extracted_xrd[:,index]=buffer
            print('iteration', index)
            
            if index == 1: # Plotting the first Pilatus image only, to check its quality
                sc = MplCanvas(self, width=5, height=4, dpi=100)
                sc.axes.pcolormesh(self.x, self.y, z, cmap='jet')
                sc.axes.plot([self.Cen_x, max(self.x)],[self.Cen_y, max(self.y)],'w')
                sc.axes.plot([self.Cen_x, max(self.x)],[self.Cen_y, min(self.y)],'w')
                sc.axes.set_xlim([0,max(self.x)]); sc.axes.set_ylim([0,max(self.y)]);

                self.setCentralWidget(sc)
                toolbar = NavigationToolbar(sc, self)
                layout = QtWidgets.QVBoxLayout()
                layout.addWidget(toolbar)
                layout.addWidget(sc)

                widget = QtWidgets.QWidget()
                widget.setLayout(layout)
                self.setCentralWidget(widget)
                self.show()
                
        self.msg = QMessageBox()
        self.msg.setText("Done!")
        x = self.msg.exec_()
        
    def plot_xrd(self): # Step 4: 2theta calibration and plotting XRD
        
        import matplotlib.cm as cm
        
        #An elementary calibration based on the x and y values of 4 peaks, to convert the x-axis to 2theta. May not always be 100% accurate but will be close and will allow the user to visualise the full spectrum and specify their own window for thte subsequent peak intensity extraction
        x1 = [214, 337, 930, 808]
        y = [24.988, 29.383, 51.266, 46.836] 

        coeff = np.polyfit(x1,y,1)
        self.x2 = np.polyval(coeff,self.x)
        
        # converting to d and plotting all XRD in terms of d
        for i in range(self.n_energypoints):
                    self.d = (two_d(self.x2, self.ene[i])) / 2
        
        for i in y:
            sc = MplCanvas(self, width=5, height=4, dpi=100)
            sc.axes.plot(self.d,self.extracted_xrd)
            sc.axes.set_xlabel("d-spacing / Å")
            sc.axes.set_ylabel("Diffracted intensity")
            sc.axes.grid(True)
            self.setCentralWidget(sc)
            
            toolbar = NavigationToolbar(sc, self)
            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(toolbar)
            layout.addWidget(sc)

            widget = QtWidgets.QWidget()
            widget.setLayout(layout)
            self.setCentralWidget(widget)
            self.alignment_factor = 0 # optional alignment factor to move the XRD to be in the desired d-range
            
            plt.plot(self.d + self.alignment_factor, self.extracted_xrd)
            plt.xlabel("d spacing / Å")
            plt.ylabel("Diffracted Peak Intensity")

        plt.show()
        self.show()
        
        # plotting all XRD in 2theta
        for i in y:
            plt.plot(self.x2, self.extracted_xrd)
            plt.xlabel("Angle / 2θ")
            plt.ylabel("Diffracted Peak Intensity")
            
        plt.show()
        
        np.savetxt(str(self.filename) + '_' + str(self.dhkl) + '_extracted_XRD',np.column_stack((self.x2,self.extracted_xrd)))
        print('done!')
        
    def exafs_file(self): #  Step 3: Importing accompanying XAS

        filename = QFileDialog.getOpenFileName(MainWindow, 'Open File')
        filename = filename[0]
        
        energies = []
        file_names = [] #this will be a list with n_energypoints pilatus file names
        I_0 = []
        exafs = []
        with open(filename, "r") as mythen_file: #opening the XAS file called mainfile and appending the energies and corresponding file names, as well as the I_0 as 3 lists
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

        # extra layer of security to ensure arrays are always the same size 
        self.I_0 = I_0[0:self.n_energypoints]
        self.exafs = exafs[0:self.n_energypoints]
        self.ene = ene[0:self.n_energypoints]
        
        x = self.msg.exec_()
        
    def plot_exafs(self):  # Step 3: Importing accompanying XAS
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.plot(self.ene,self.exafs, color = 'black')
        sc.axes.set_xlabel("Energy / eV")
        sc.axes.set_ylabel("Intensity")
        sc.axes.grid(True)
        self.setCentralWidget(sc)
        
        toolbar = NavigationToolbar(sc, self)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(sc)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.show()
        
        plt.plot(self.ene, self.exafs, color = 'black')
        plt.xlabel("Energy / eV")
        plt.ylabel("μ(E)")
        plt.show()
        
        np.savetxt(str(self.filename) + '_' + str(self.dhkl) + '_exafs',np.column_stack((self.ene,self.exafs,self.I_0)))
        
    def get_peak_areas(self): # step 5: extracting intensities from a given peak
        
    
        n = len(sys.argv)
        self.dhkl = self.d_label.text()
        self.dhkl = [self.dhkl]
        
        for dhkl in self.dhkl: # take a given d value as specified by system argument, and use it to obtain a series of 'max_y' values within a given angle range
            self.alignment_factor = float(self.window_text.text())
            area_arr  = []
            a = []
            for i in range(self.n_energypoints):
                    print(i)
                    I = self.extracted_xrd[:,i]
                    x = self.x2
                    dhkl = float(dhkl)
                    self.d = ((two_d(x, self.ene[i])) / 2) + self.alignment_factor #converts two theta to d using the two_d function
                    
                    dmin = float(self.d_min.text())
                    dmax = float(self.d_max.text()) #user should define these based on their desired window 
                    
                    index_d = np.where((self.d >= dmin) & (self.d <= dmax))  # stores the number of values of d, that is, the index of d, between the specified range, i.e. the 12th, 13th, 14th value
                    d_select = self.d[index_d[0]] 
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
                        
                    # Plotting within the specified d window
                    sc = MplCanvas(self, width=5, height=4, dpi=100)
                    sc.axes.plot(d_select,y_select, color = 'black')
                    sc.axes.set_xlabel("d spacing")
                    sc.axes.set_ylabel("Intensity")
                    sc.axes.grid(True)
                    self.setCentralWidget(sc)
                    
                    toolbar = NavigationToolbar(sc, self)
                    layout = QtWidgets.QVBoxLayout()
                    layout.addWidget(toolbar)
                    layout.addWidget(sc)

                    widget = QtWidgets.QWidget()
                    widget.setLayout(layout)
                    self.setCentralWidget(widget)
                    self.show()

                    mod = PseudoVoigtModel(prefix = 'pv_')
                    params = mod.make_params(pv_amplitude = area_guess, pv_height = amp, pv_center=mean, pv_sigma=sigma)

                    mod_lin = LinearModel(prefix= 'lin_')
                    params.update( mod_lin.guess(y_select, x=d_select))

                    # sum components to make a composite model (add more if needed)
                    model  =  mod + mod_lin
                    out = model.fit(y_select, params, x=d_select)
                    print(out.fit_report())
                    comps = out.eval_components(x=d_select)
                            
                    res = (out.best_fit) 
                    area = (trapz(comps['pv_'])) #trapezoidal integration of the pv component
                    a.append(out.params.get('pv_amplitude').value)  
                    area_arr.append(area)
                    
                    if i == 1: #plots d in the specified window, with the data subtraction, for the 1st dataset
                        plt.plot(d_select, y_select, color = 'black', label = 'experimental data')
                        plt.plot(d_select, comps['pv_'], color = 'blue',label='Pseudo-Voigt component')
                        plt.plot(d_select, comps['exp_'], '--',color = 'red', label='BKG component')
                        plt.plot(d_select,res, '--', color = 'green', label = 'total fit')
                        plt.xlabel("d-spacing / Ang.")
                        plt.ylabel("Diffracted peak intensity")
                        plt.legend()
                        plt.show()
                        
                        sc.axes.plot(d_select,y_select)
                        sc.axes.plot(d_select,comps['pv_'])
                        sc.axes.plot(d_select,comps['exp_'])
                        sc.axes.plot(d_select,res)
                        self.show()
                
            area_arr = np.array(area_arr)
            area_norm = norm_I(area_arr,self.I_0)
            a_norm = norm_I(a,self.I_0)

            y_new = abs((area_norm)) / max(area_norm)
            a_new = abs((a_norm))

            mask = np.isnan(y_new)
            idx = np.where(~mask,np.arange(mask.shape[0]),0)
            np.maximum.accumulate(idx, out=idx)
            self.y_new = y_new[idx] #gets rid of any values less than 0
            
            self.edge = self.E_label.text() #user specified edge
            self.edge = float(self.edge)
            
            y_new_shift = self.y_new

            min_y_new_shift = min(y_new_shift)
            index_y_new_shift = np.where(y_new_shift <= min_y_new_shift)
            y_new_shift_select = self.ene[index_y_new_shift[0]]
            print(y_new_shift_select)

            self.difference = y_new_shift_select - self.edge
            print(self.difference)
            
            self.y_new_shift = y_new_shift
            self.ene = self.ene - self.difference  
            
            # plotting and saving final result of raw DAFS spectrum
            sc = MplCanvas(self, width=5, height=4, dpi=100)
            sc.axes.plot(self.ene,a_new, color = 'black')
            sc.axes.set_xlabel("Energy / eV")
            sc.axes.set_ylabel("Intensity")
            sc.axes.grid(True)
            self.setCentralWidget(sc)
            
            toolbar = NavigationToolbar(sc, self)
            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(toolbar)
            layout.addWidget(sc)

            widget = QtWidgets.QWidget()
            widget.setLayout(layout)
            self.setCentralWidget(widget)
            self.show()
            
            plt.plot(self.ene, self.y_new_shift,  color = 'black')
            plt.xlabel("Energy / eV")
            plt.ylabel("Diffracted Intensity")
            plt.show()
            
            np.savetxt(str(self.filename) + '_' + str(self.dhkl) + '_results',np.column_stack((self.ene,self.I_0,area_norm,y_new)))
            
    def first_guess_model(self): # Beginning the model, defining the model fit function 'intensity'
        
        #Calculating sin theta for model input
        self.dhkl = self.d_label.text()
        self.dhkl = float(self.dhkl)
        sin_theta = []
        for self.i in range(len(self.ene)):
            sin_theta.append( (12938. / (2* self.dhkl * self.ene[self.i])))
        self.sin_theta = np.asarray(sin_theta)
        
        def intensity(en, phi, beta, t, exafs, sin_theta, scale=1, slope=0, offset=0, fprime=-1, fsec=1):
            costerm = (np.cos((phi)) + (beta*fprime))
            sinterm = (np.sin((phi)) + (beta*fsec))
            return scale * ((costerm**2 + sinterm**2)) * (1 - e**((-2*exafs*t)/sin_theta)) / (2*exafs) + offset
        
        self.z_abs = self.z_label.text()
        self.z_abs = int(self.z_abs)
        self.edge = self.E_label.text()
        self.edge = float(self.edge)
        
        self.exafs = self.exafs[0:len(self.y_new_shift)]
        self.ene = self.ene[0:len(self.y_new_shift)]
        
        self.dkk=diffkk(self.ene, self.exafs, z= self.z_abs, edge='K', mback_kws={'e0':self.edge, 'order': 4})
        self.dkk.kk()
  
        imodel = Model(intensity, independent_vars=['en', 'fprime', 'fsec', 'exafs', 'sin_theta'])
        params = imodel.make_params(scale=1, offset=0, slope=0, beta=0.1, phi= math.pi, t = 1) 
        # we can constrain parameters as desired. E.G.:
        params['scale'].min = 0 

        result = imodel.fit(self.y_new_shift, params, en=self.ene_shift, fprime=self.dkk.f1, fsec=self.dkk.f2, exafs = self.exafs, sin_theta = self.sin_theta)
        print(result.fit_report())

        self.phi = result.params.get('phi').value 
        self.beta = result.params.get('beta').value
        self.I0 = result.params.get('scale').value
        self.Ioff = result.params.get('offset').value
        self.slope = result.params.get('slope')
        self.t = result.params.get('t')

        # plot the first guess fit and the experimental data
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.plot(self.ene, (result.best_fit), '--', label='best fit (lmfit)', color = 'blue')
        sc.axes.plot(self.ene, self.y_new_shift, label='exp. data', color = 'red')
        sc.axes.set_xlabel("Energy / eV")
        sc.axes.set_ylabel("Intensity")
        sc.axes.grid(True)
        sc.axes.legend()
        self.setCentralWidget(sc)
        
        toolbar = NavigationToolbar(sc, self)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(sc)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.show()

        np.savetxt(str(self.filename) + '_' + str(self.dhkl) + '_model_1st_guess_and_new_abscorr',np.column_stack((self.ene,self.y_new_shift,result.best_fit,self.abscorr)))
        
    def first_guess_f1(self): #calculating f' and f" based on above fit
        
        def f1_guess(fsec, I, I0, phi, beta, Ioff, abscorr):
            f1_guess = (1/beta) * +(np.sqrt(((((I)-Ioff))/(I0*abscorr)) -((math.sin(phi)) +(beta*fsec))**2)  - (math.cos(phi)))
            return f1_guess
        
        self.f1_minus = f1_guess(self.dkk.f2, self.y_new_shift, self.I0, self.phi, self.beta, self.Ioff, self.abscorr)
        
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.plot(self.ene_shift, self.f1_minus, label='first guess', color = 'black')
        sc.axes.plot(self.dkk.energy, self.dkk.f1, '--', label='atomic fprime', color = 'black')
        sc.axes.plot(self.dkk.energy, self.dkk.fp, '--', label='molecular fprime', color = 'black')
        
        sc.axes.set_xlabel("Energy / eV")
        sc.axes.set_ylabel("Intensity")
        sc.axes.grid(True)
        sc.axes.legend()
        self.setCentralWidget(sc)
        
        toolbar = NavigationToolbar(sc, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(sc)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        plt.show()
        self.show()
        
        np.savetxt(str(self.filename) + '_' + str(self.dhkl) + '_f1',np.column_stack((self.ene,self.f1_minus, self.dkk.f1, self.dkk.fp)))

    def first_guess_f2(self):
        ene = np.array(self.ene)
        ene_dash = np.array(self.ene)
        f1_minus = self.f1_minus

        i = (len(self.y_new_shift) - 1)  
        
        # KK transforming to produce f" from f'
        f2KK = lambda ene_dash : (f1_minus - self.dkk.f1) / ((ene_dash-1)**2 - (ene[i])**2) 
        print('f2KK : ', f2KK)
        f2KK_arr = []
        f2KK_arr, err = (quad_vec(f2KK,(ene_dash[0]),(ene_dash[-1])))
        print('f2kk_arr : ', f2KK_arr)
        self.fsecond_KK = (self.dkk.f2 - ((2*self.ene/math.pi) * f2KK_arr))

        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.plot(self.ene,self.fsecond_KK, label = 'new f2')
        sc.axes.plot(self.ene,self.dkk.f2, label = 'atomic f2')
        sc.axes.plot(self.ene,self.f1_minus, label = 'new f1')
        sc.axes.plot(self.ene,self.dkk.f1, label = 'atomic f1')
        
        sc.axes.plot(self.ene,self.dkk.fp)
        sc.axes.plot(self.ene,self.dkk.fpp)
        
        sc.axes.set_xlabel("Energy / eV")
        sc.axes.set_ylabel("Intensity")
        sc.axes.grid(True)
        sc.axes.legend()
        self.setCentralWidget(sc)
        
        toolbar = NavigationToolbar(sc, self)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(sc)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.show()
        
        np.savetxt(str(self.filename) + '_' + str(self.dhkl) + '_fsec',np.column_stack((self.ene,self.fsecond_KK, self.dkk.f2, self.dkk.fpp)))

    def iterate(self): #can be run as many times as desired until model converges.
        imodel = Model(intensity, independent_vars=['en', 'fprime', 'fsec', 'abscorr'])
        params = imodel.make_params(scale=self.I0, offset=self.Ioff, slope=0, beta=self.beta, phi=self.phi)
        params['scale'].min = 0

        result = imodel.fit(self.y_new_shift, params, en=self.ene, fprime=self.f1_minus, fsec=self.fsecond_KK, abscorr = self.abscorr)
        print(result.fit_report())

        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.plot(self.ene, (result.best_fit), '--', label='best fit (lmfit)', color = 'blue')
        sc.axes.plot(self.ene, self.y_new_shift, label='best fit (lmfit)', color = 'red')
        sc.axes.set_xlabel("Energy / eV")
        sc.axes.set_ylabel("Intensity")
        sc.axes.grid(True)
        self.setCentralWidget(sc)
        
        toolbar = NavigationToolbar(sc, self)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(sc)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.show()

        self.phi = result.params.get('phi').value 
        self.beta = result.params.get('beta').value
        self.I0 = result.params.get('scale').value
        self.Ioff = result.params.get('offset').value
        self.slope = result.params.get('slope')
        print(' I0 = ',self.I0, '\n','phi = ', self.phi, '\n','beta =',self.beta, '\n','Ioff =', self.Ioff, '\n','energy dependence =', self.slope)

        self.f1_new = f1_guess(self.fsecond_KK, self.y_new_shift, self.I0, self.phi, self.beta, self.Ioff, self.abscorr)
        i = (len(self.y_new_shift) - 1) 

        ene = np.array(self.ene)
        ene_dash = np.array(self.ene)
        f1_minus = self.f1_new
        dkk_f1 = self.dkk.f1
        
        f2KK = lambda ene_dash : ((f1_minus - dkk_f1)/ ((ene_dash-1)**2 - (ene[i])**2))

        f2KK_arr = []
        f2KK_arr, err = (quad_vec(f2KK,(ene_dash[0]),(ene_dash[-1])))

        self.fsecond_KK = (self.dkk.f2 - ((2*self.ene/math.pi) * f2KK_arr)) 

        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.plot(self.ene,self.fsecond_KK, label = 'new f2')
        sc.axes.plot(self.ene,self.dkk.f2, label = 'atomic f2')
        sc.axes.plot(self.ene,f1_minus, label = 'new f1')
        sc.axes.plot(self.ene,self.dkk.f1, label = 'atomic f1')
        
        sc.axes.set_xlabel("Energy / eV")
        sc.axes.set_ylabel("Intensity")
        sc.axes.grid(True)
        sc.axes.legend()
        self.setCentralWidget(sc)
        toolbar = NavigationToolbar(sc, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(sc)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.show()

    def save(self): # saves the final result as a .txt file
        np.savetxt(str(self.dhkl) + 'f"_final_iteration.txt',np.column_stack((self.ene,self.fsecond_KK,self.dkk.f2,self.f1_minus,self.dkk.f1)))

stylesheet = """

QMainWindow {background-color: light grey; }

"""

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(stylesheet)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
