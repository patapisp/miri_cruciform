import numpy as np
from scipy.interpolate import interp1d
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
import sys

from diffractio import degrees, mm, plt, sp, um, np
from diffractio.scalar_fields_XY import Scalar_field_XY
from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ
from diffractio.utils_drawing import draw_several_fields
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY

#from diffractio.utils_math import ndgrid

import matplotlib.cm as cm

import time
import os
os.chdir("/Users/polychronispatapis/Documents/Projects/miripsf/miri_cruciform")

# IR absorption model

def alpha_coeff(lam):
    return 102*(lam/7)**2 #1/cm

def IR_absorption(I0, lam, t=35*1E-4):
    return (1-np.exp(-alpha_coeff(lam)*t))*I0

def PixelGrid_absorption():
    return (1-(27**2-25**2)/27**2)

def lrs_detector_dispersion(wavelength):
    return -3*wavelength**2 + 16.96*wavelength + 69.32

wav_data,reflectance = np.genfromtxt('data/SW_ARcoat_reflectance.txt', skip_header=4, usecols=(0, 1), delimiter=',', unpack=True)
Rl = interp1d(wav_data,reflectance)
wavs = np.linspace(2., 26, num=100)

def fresnel_r_from_angles(n1, n2, theta):
    # theta: incidence angle inside medium n1
    # returns complex r_s, r_p
    # Snell
    sin_t = np.minimum(1, n1/n2 * np.sin(theta))
    # Handle TIR: when argument >1, set cos_t to imaginary
    cos_i = np.cos(theta)
    arg = 1 - (n1/n2*np.sin(theta))**2
    cos_t = np.sqrt(arg + 0j)  # allow complex

    r_s = (n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t)
    r_p = (n2*cos_i - n1*cos_t) / (n2*cos_i + n1*cos_t)
    return r_s, r_p

def fresnel_reflect_field(E, wavelength, n1, n2, dx, polarization='unpolarized'):
    Ny, Nx = E.shape
    k0 = 2*np.pi/wavelength

    E_k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E)))

    fx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, d=dx))
    FX, FY = np.meshgrid(fx, fy)
    kx = 2*np.pi*FX
    ky = 2*np.pi*FY
    k_parallel = np.sqrt(kx**2 + ky**2)

    sin_theta = np.clip(k_parallel / (k0 * n1), -1, 1)
    theta = np.arcsin(sin_theta)

    r_s, r_p = fresnel_r_from_angles(n1, n2, theta)

    if polarization == 'unpolarized':
        Es_k = E_k/np.sqrt(2)
        Ep_k = E_k/np.sqrt(2)
        E_ref_k = Es_k*r_s + Ep_k*r_p
    elif polarization == 's':
        E_ref_k = E_k * r_s
    elif polarization == 'p':
        E_ref_k = E_k * r_p
    else:
        raise ValueError("polarization must be 'unpolarized', 's', or 'p'")

    E_ref = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E_ref_k)))
    return E_ref


class MIRICruciform:
    """
    Code to simulate the MIRI cruciform pattern using diffraction and fresnel propagation
    """

    def __init__(self, mode="IMA", simsize=2048, pupilfile="JWpupil_segments_1024x1024.npy", filterpath="./"):
        self.mode = mode
        self.specR = {"LRS":50, "MRS":3000}
        self.focal_ratios = {"IMA": 7.0, "MRS": 3.0, "LRS-SLTSS": 7.0, "LRS-SLT": 7.0}
        self.diameter_mode = {"IMA": 1.0, "MRS":1.0, "LRS-SLTSS": 1.0, "LRS-SLT": 1.0}
        self.length_mode = {"IMA": 2.5, "MRS": 1.0, "LRS-SLTSS": 2.5, "LRS-SLT": 2.5}
        self.rotation = {"IMA": -4.8, "MRS": -7.7, "LRS-SLTSS": -4.8, "LRS-SLT": -4.8}
        # load JWST pupil
        jwst_pupil = np.load(pupilfile)
        padlen = int((simsize - np.shape(jwst_pupil)[0])/2)
        self.jwst_pupil = np.pad(jwst_pupil, padlen, mode='constant')
        # plt.figure()
        # plt.imshow(jwst_pupil, origin="lower")
        self.wavelength = 5.0 * um
        self.num_pixels = simsize
        self.diameter = self.diameter_mode[self.mode] * mm
        self.length = self.length_mode[self.mode] * mm
        self.focal = self.focal_ratios[self.mode] * mm
        self.x0 = np.linspace(-self.length / 2, self.length / 2, self.num_pixels)
        self.y0 = np.linspace(-self.length / 2, self.length / 2, self.num_pixels)
        self.distance_pupil_ar = self.focal_ratios[self.mode] - 0.2
        self.filterfiles = {"F560W":filterpath+"JWST_MIRI.F560W.dat",
                            "F770W": filterpath+"JWST_MIRI.F770W.dat",
                            "monochromatic": None}
        self.dispersion_angle = 0.

    def intialise_wavefront(self):
        self.dispersion(target_wave=self.wavelength, mode=self.mode)
        self.u0 = Scalar_source_XY(x=self.x0, y=self.y0, wavelength=self.wavelength)
        self.t0 = Scalar_mask_XY(x=self.x0, y=self.y0, wavelength=self.wavelength)
        self.u0.plane_wave(theta=0.0 * degrees, phi=self.dispersion_angle * degrees)
        self.pupil = Scalar_mask_XY(x=self.x0, y=self.y0, wavelength=self.wavelength)
        self.pupil.u = self.jwst_pupil * np.exp(1j * np.zeros_like(self.jwst_pupil))
        self.pupil.rotate(self.rotation[self.mode] * np.pi / 180)
        self.t0.lens(r0=(0 * um, 0 * um), radius=(self.diameter / 2, self.diameter / 2), focal=(self.focal, self.focal))
        self.t0 = self.t0 * self.pupil
        self.u1 = self.u0 * self.t0
        return

    def detector_grid(self, fill=0.85):
        # detector grid
        self.dg = Scalar_mask_XY(x=self.x0, y=self.y0, wavelength=self.wavelength)
        self.dg.grating_2D(period=27 * um,
                      a_min=1.,
                      a_max=1.,
                      phase=np.pi,
                      r0=(0, 0),
                      fill_factor=fill)
        return

    def incidence_angle(self, angle=0.):
        # tilt
        self.tilt = Scalar_mask_XY(x=self.x0, y=self.y0, wavelength=self.wavelength)
        t = 2 * np.pi * np.sin(angle * np.pi / 180) * (self.y0 + self.length / 2) * 3.4 / (self.wavelength)
        tilt_phase = np.repeat(t, self.num_pixels).reshape(self.num_pixels, self.num_pixels)
        self.tilt.u = np.ones((self.num_pixels, self.num_pixels)) * np.exp(1j * tilt_phase)
        return

    def internal_total_reflection(self, tr_radius=140.0):
        self.Rmask = Scalar_mask_XY(self.x0, self.y0, wavelength=self.wavelength)
        self.Rmask.u = np.ones_like(self.Rmask.u)
        Rmask2 = Scalar_mask_XY(self.x0, self.y0, wavelength=self.wavelength)
        Rmask2.circle(
            r0=(0 * um, 0 * um), radius=(tr_radius * um, tr_radius * um), angle=0 * degrees)
        self.Rmask -= Rmask2
        return
    
    def calculate_propagation_angle(self, field, distance):
        # Apply a window function to the field distribution
        
        windowed_field = field * np.hanning(field.shape[0])[:, np.newaxis]

        # Perform 2D Fourier transform of the windowed field at z = 0
        field_spectrum = fft2(windowed_field)

        # Define the wave number
        k = 2 * np.pi / self.wavelength

        # Generate frequency coordinates
        nx, ny = field.shape
        dx = self.wavelength * distance / (nx - 1)
        dy = self.wavelength * distance / (ny - 1)
        fx = np.fft.fftfreq(nx, dx)
        fy = np.fft.fftfreq(ny, dy)
        kx, ky = np.meshgrid(fx, fy, indexing='ij')

        # Calculate the longitudinal component of the wave vector
        kz = np.sqrt(np.maximum(0, k**2 - kx**2 - ky**2))

        # Calculate the propagation angle
        propagation_angle = np.arctan2(np.sqrt(kx**2 + ky**2), (k * kz))

        return propagation_angle


    def dispersion(self, target_wave, mode="IMA"):
        if mode == "MRS":
            self.dispersion_angle = 0.0
        elif mode in ["LRS-SLTSS", "LRS-SLT"]:
            fl = self.focal_ratios[mode]*1e6 # in um            
            central_wave = 8.4*um
            self.dispersion_angle = ((target_wave*um-central_wave)*lrs_detector_dispersion(target_wave)*25*um/fl)/degrees
        else:
            self.dispersion_angle = 0.0
        return

    def layer_absorption(self):
        self.A = Scalar_mask_XY(x=self.x0, y=self.y0, wavelength=self.wavelength)
        self.T = Scalar_mask_XY(x=self.x0, y=self.y0, wavelength=self.wavelength)
        factor = IR_absorption(1.0, lam=self.wavelength)
        double_pass = (factor+factor*(1-factor)) #*PixelGrid_absorption()
        self.A.u = np.ones_like(self.A.u)*double_pass
        self.T.u = np.ones_like(self.T.u)*(1-double_pass)
        return


    def monochromatic_webbpsf(self, wavelength=5.0, detector_angle=0):
        self.wavelength = wavelength * um
        self.intialise_wavefront()

        z0 = np.linspace(0 * mm, self.distance_pupil_ar * mm, 16)
        u2 = Scalar_field_XYZ(x=self.x0, y=self.y0, z=z0, wavelength=wavelength, n_background=1.)
        u2.incident_field(self.u1)
        u2.clear_field()
        u2.WPM()
        u2 = u2.to_Scalar_field_XY(iz0=-1)
        self.incidence_angle(detector_angle) # apply detector tilt
        u2 = u2 * self.tilt

        z0 = np.linspace(self.distance_pupil_ar * mm, (self.distance_pupil_ar+0.5) * mm, 16)
        u3 = Scalar_field_XYZ(x=self.x0, y=self.y0, z=z0, wavelength=self.wavelength, n_background=3.4)
        u3.incident_field(u2)

        u3.clear_field()
        u3.WPM()
        u3 = u3.to_Scalar_field_XY(iz0=-1)
        return u3 # .intensity()



    def monochromatic_cruciform(self, wavelength=5.0, a1=1., a2=0.6, a3=0.45, detector_angle=0.0, tr_radius=140.0,
                                verbose=True, return_intensity=True):
        self.wavelength = wavelength * um
        self.intialise_wavefront()
        self.layer_absorption()

        z0 = np.linspace(0 * mm, self.distance_pupil_ar * mm, 16)
        u2 = Scalar_field_XYZ(x=self.x0, y=self.y0, z=z0, wavelength=wavelength, n_background=1.)
        u2.incident_field(self.u1)
        u2.clear_field()
        u2.WPM()
        u2 = u2.to_Scalar_field_XY(iz0=-1)
        self.incidence_angle(detector_angle) # apply detector tilt
        u2 = u2 * self.tilt

        z0 = np.linspace(self.distance_pupil_ar * mm, (self.distance_pupil_ar+0.5) * mm, 16)
        u3 = Scalar_field_XYZ(x=self.x0, y=self.y0, z=z0, wavelength=self.wavelength, n_background=3.4)
        u3.incident_field(u2)

        u3.clear_field()
        u3.WPM()
        u3 = u3.to_Scalar_field_XY(iz0=-1)
        z0 = np.linspace((self.distance_pupil_ar+0.5) * mm, (self.distance_pupil_ar+1) * mm, 16)
        u4ar = Scalar_field_XYZ(x=self.x0, y=self.y0, z=z0, wavelength=self.wavelength, n_background=3.4)
        self.detector_grid()
        u4ar.incident_field(u3 * self.T * self.dg)
        u4ar.clear_field()
        u4ar.WPM()

        #self.internal_total_reflection(tr_radius=tr_radius)
        _tmp = u4ar.to_Scalar_field_XY(iz0=-1)
        dx = float(self.x0[1] - self.x0[0])
        E_ref = fresnel_reflect_field(_tmp.u, wavelength=self.wavelength, n1=3.4, n2=1.0, dx=dx, polarization='unpolarized')
        _ref_field = Scalar_field_XY(x=self.x0, y=self.y0, wavelength=self.wavelength)
        _ref_field.u = E_ref
        z0 = np.linspace( (self.distance_pupil_ar+1) * mm, (self.distance_pupil_ar+1.5) * mm, 16)
        u4 = Scalar_field_XYZ(x=self.x0, y=self.y0, z=z0, wavelength=self.wavelength, n_background=3.4)
        #u4.incident_field(u4ar.to_Scalar_field_XY(iz0=-1) * self.Rmask)
        u4.incident_field(_ref_field)

        u4.clear_field()
        u4.WPM()
        u4 = u4.to_Scalar_field_XY(iz0=-1)

        z0 = np.linspace((self.distance_pupil_ar + 1.5) * mm, (self.distance_pupil_ar + 2) * mm, 16)
        u5ar = Scalar_field_XYZ(x=self.x0, y=self.y0, z=z0, wavelength=self.wavelength, n_background=3.4)
        self.detector_grid()
        u5ar.incident_field(u4 * self.T * self.dg)
        u5ar.clear_field()
        u5ar.WPM()

        #self.internal_total_reflection(tr_radius=tr_radius)
        _tmp = u5ar.to_Scalar_field_XY(iz0=-1)
        dx = float(self.x0[1] - self.x0[0])
        E_ref = fresnel_reflect_field(_tmp.u, wavelength=self.wavelength, n1=3.4, n2=1.0, dx=dx, polarization='unpolarized')
        _ref_field = Scalar_field_XY(x=self.x0, y=self.y0, wavelength=self.wavelength)
        _ref_field.u = E_ref
        z0 = np.linspace((self.distance_pupil_ar + 2.) * mm, (self.distance_pupil_ar + 2.5) * mm, 16)
        u5 = Scalar_field_XYZ(x=self.x0, y=self.y0, z=z0, wavelength=wavelength, n_background=3.4)
        #u5.incident_field(u5ar.to_Scalar_field_XY(iz0=-1) * self.Rmask)
        u5.incident_field(_ref_field)
        u5.clear_field()
        u5.WPM()
        u5 = u5.to_Scalar_field_XY(iz0=-1)

        # u3.normalize()
        # u4.normalize()
        # a2 *= IR_absorption(1.0, 5.0)/IR_absorption(1.0, wavelength)
        # a3 *= IR_absorption(1.0, 5.0) / IR_absorption(1.0, wavelength)
        if return_intensity:
            return (self.A * u3).intensity() + (self.A * u4).intensity() + (self.A * u5).intensity(), \
               u3.intensity(), u4.intensity(), u5.intensity()
        else: 
            return (self.A * u3),  (self.A * u4),  (self.A * u5)

    #(self.A * u3).intensity() + (self.A * u4).intensity() + (self.A * u5).intensity()

    def MIRIFilter(self, wavelength_points=10, tr_radius=140, detector_angle=0.0, filter="F560W"):
        # load filter profile
        w, f = np.array(list(zip(*np.loadtxt(self.filterfiles[filter]))))
        ind = np.linspace(5, len(w)-5, num=wavelength_points, dtype=int)
        wav_array = w[ind]/10000
        transmission = f[ind]
        n = len(wav_array)
        ufc = 0
        uf = 0
        uc1 = 0
        uc2 = 0
        for i, wav in enumerate(wav_array):
            j = (i + 1) / n
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j))
            sys.stdout.flush()
            utotal, uwebb, ucruci1, ucruci2 = self.monochromatic_cruciform(wavelength=wav, detector_angle=detector_angle,
                                                                  tr_radius=tr_radius)
            ufc += utotal  # transmission[i]* account for filter transmission
            uf += uwebb
            uc1 += (ucruci1)
            uc2 += (ucruci2)
        return [ufc, uf, uc1, uc2]
    
    def LRSsim(self, wavelengths=np.linspace(5,12,num=10), detector_angle=0.0, tr_radius=0.0):
        n = len(wavelengths)
        ufc = 0
        uf = 0
        uc1 = 0
        uc2 = 0
        for i, wav in enumerate(wavelengths):
            j = (i + 1) / n
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j))
            sys.stdout.flush()
            utotal, uwebb, ucruci1, ucruci2 = self.monochromatic_cruciform(wavelength=wav, detector_angle=detector_angle,
                                                                  tr_radius=tr_radius)
            ufc += utotal  # transmission[i]* account for filter transmission
            uf += uwebb
            uc1 += (ucruci1)
            uc2 += (ucruci2)
        return [ufc, uf, uc1, uc2]
    
    def plot_cruciform(self, components, filter, savepath=None, **kwargs):
        titles = [f"MIRI PSF: {filter}", "Webb PSF", "Cruciform 3rd pass", "Cruciform 5th pass"]
        fig, ax = plt.subplots(1, len(components), figsize=(12, 6))
        ax = np.array(ax).flatten()
        im = 0
        for i, c in enumerate(components):
            im = ax[i].imshow(c, origin="lower", **kwargs)
            fig.colorbar(im, format="$%.2f$")

        # fig.subplots_adjust(right=0.85)
        # cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        # fig.colorbar(im, cax=cbar_ax)
        #
        # if savepath is not None:
        #     plt.savefig(savepath+f"MIRI_cruciformsim_{filter}.fits")
        # return

    def save_simulation(self, components, path="/Users/polychronispatapis/Box/MIRI-COMM-Team/Sandbox/patapisp/DiffractionSimulations/simulations/", savename="MIRI_cruciformSim.fits", metadata=None):
        primary_hdu = fits.PrimaryHDU(data=components[0])
        hdul = fits.HDUList(hdus=[primary_hdu])
        if len(components) >1:
            for i in np.arange(1, len(components)):
                hdul.append(fits.ImageHDU(data=components[i]))

        if metadata is not None:
            for k, v in metadata.items():
                hdul[0].header[k] = v
        if "filter" in metadata.keys():
            savename = f"MIRI_cruciformsim_{metadata['filter']}_deta{metadata['dettilt']}deg_TIR{metadata['TIR']}um.fits"
        hdul.writeto(path + savename, overwrite=True)

    def runsim(self, filter="F560W", wavelength_points=10, tr_radius=150, detector_angle=0.0, plot=True, save=True,
               savepath="./simulations/"):
        if 'LRS' in filter:
            print(f"Running Simulation for filter: {filter}")
            psfs = self.LRSsim()
        elif filter in self.filterfiles.keys():
            print(f"Running Simulation for filter: {filter}")
            psfs = self.MIRIFilter(wavelength_points=wavelength_points, tr_radius=tr_radius, filter=filter,
                                   detector_angle=detector_angle)
        else: 
            if float(filter) < 4 or float(filter) >25:
                print(f"Value for wavelength={filter} not accepted")
                raise ValueError

            psfs = self.monochromatic_cruciform(wavelength=float(filter), detector_angle=detector_angle,
                                                                  tr_radius=tr_radius)
        if save:
            self.save_simulation(components=psfs, path=savepath, metadata={"filter": filter, "dettilt":detector_angle,
                                                                           "TIR": tr_radius})
        try:
            if plot:
                self.plot_cruciform(components=psfs, filter=filter, savepath=savepath, norm=LogNorm(vmin=1E-3, vmax=100), cmap="viridis")
        except:
            pass
        return


if __name__ == "__main__":
    mr = MIRICruciform(mode='LRS-SLTSS', 
                       pupilfile="/Users/polychronispatapis/Documents/Projects/miripsf/miri_cruciform/data/JWpupil_segments_1024x1024.npy", 
                       filterpath="/Users/polychronispatapis/Documents/Projects/miripsf/miri_cruciform/data/")
    # psf = mr.monochromatic_cruciform()
    # # Start a timer to keep track of runtime
    time0 = time.perf_counter()
    # mr.runsim(filter="F560W", wavelength_points=10, tr_radius=150, detector_angle=1.5)
    mr.runsim(filter="LRS-SLTSS", wavelength_points=10, tr_radius=150, detector_angle=1.5)
    # Print out the time benchmark
    time1 = time.perf_counter()
    print(f"Runtime so far: {time1 - time0:0.4f} seconds")
    # psf = mr.monochromatic_cruciform(wavelength=5.0, tr_radius=150)
    # plt.figure()
    # plt.imshow((psf[2]), origin="lower", vmax=np.max(psf[2])*0.01)
    # plt.show()
    # plt.figure()
    # plt.imshow((psf[3]), origin="lower", vmax=np.max(psf[3])*0.01)
    # plt.show()





