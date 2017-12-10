"""Class to generate simulation ICS, separated out for clarity."""
from __future__ import print_function
import os.path
import math
import subprocess
import json
#To do crazy munging of types for the storage format
import importlib
import numpy as np
import configobj
import camb
from camb import model
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
from nbodykit.lab import BigFileCatalog,FFTPower
from . import simulation
from . import cambpower
from . import utils

class SimulationICs(object):
    """
    Class for creating the initial conditions for a simulation.
    There are a few things this class needs to do:

    - Generate CAMB input files
    - Generate N-GenIC input files (to use CAMB output)
    - Run CAMB and N-GenIC to generate ICs

    The class will store the parameters of the simulation.
    We also save a copy of the input and enough information to reproduce the resutls exactly in SimulationICs.json.
    Many things are left hard-coded.
    We assume flatness.

    Init parameters:
    outdir - Directory in which to save ICs
    box - Box size in comoving Mpc/h
    npart - Cube root of number of particles
    redshift - redshift at which to generate ICs
    separate_gas - if true the ICs will contain baryonic particles. If false, just DM.
    omegab - baryon density. Note that if we do not have gas particles, still set omegab, but set separate_gas = False
    omega0 - Total matter density at z=0 (includes massive neutrinos and baryons)
    hubble - Hubble parameter, h, which is H0 / (100 km/s/Mpc)
    scalar_amp - A_s at k = 2e-3, comparable to the WMAP value.
    ns - Scalar spectral index
    m_nu - neutrino mass
    """
    def __init__(self, *, outdir, box, npart, seed = 9281110, redshift=99, separate_gas=True, omega0=0.288, omegab=0.0472, hubble=0.7, scalar_amp=2.427e-9, ns=0.97, rscatter=False, m_nu=0, nu_hierarchy='degenerate', code_class=simulation.Simulation, code_args=None):
        #This lets us safely have a default dictionary argument
        self.code_args = {}
        if code_args is not None:
            self.code_args.update(code_args)
        #Check that input is reasonable and set parameters
        #In Mpc/h
        assert box < 20000
        self.box = box
        #Cube root
        assert npart > 1 and npart < 16000
        self.npart = int(npart)
        #Physically reasonable
        assert omega0 <= 1 and omega0 > 0
        self.omega0 = omega0
        assert omegab > 0 and omegab < 1
        self.omegab = omegab
        assert redshift > 1 and redshift < 1100
        self.redshift = redshift
        assert hubble < 1 and hubble > 0
        self.hubble = hubble
        assert scalar_amp < 1e-7 and scalar_amp > 0
        self.scalar_amp = scalar_amp
        assert ns > 0 and ns < 2
        self.ns = ns
        self.rscatter = rscatter
        outdir = os.path.realpath(os.path.expanduser(outdir))
        #Make the output directory: will fail if parent does not exist
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        else:
            if os.listdir(outdir) != []:
                print("Warning: ",outdir," is non-empty")
        #Structure seed.
        self.seed = seed
        #Baryons?
        self.separate_gas = separate_gas
        #If neutrinos are combined into the DM,
        #we want to use a different CAMB transfer when checking output power.
        self.separate_nu = False
        self.m_nu = m_nu
        self.nu_hierarchy = nu_hierarchy
        self.outdir = outdir
        defaultpath = os.path.dirname(__file__)
        #Default GenIC paths
        self.genicdefault = os.path.join(defaultpath,"ngenic.param")
        self.genicout = "_genic_params.ini"
        #Executable names
        self.genicexe = "N-GenIC"
        #Number of files per snapshot
        #This is chosen to give a reasonable number and
        #a constant number of particles per file.
        self.numfiles = int(np.max([2,self.npart**3//2**24]))
        #Class with which to generate ICs.
        self.code_class_name = code_class
        #Format in which to output initial conditions: derived from Simulation class.
        self.icformat = code_class.icformat
        assert 4 >= self.icformat >= 2

    def cambfile(self):
        """Generate the IC power spectrum using pyCAMB."""
        #Load CAMB file using ConfigObj
        pars = camb.CAMBparams()
        #Set the neutrino density and subtract it from omega0
        omeganuh2 = self.m_nu/93.14
        omch2 = (self.omega0 - self.omegab)*self.hubble**2 - omeganuh2
        ombh2 =  self.omegab*self.hubble**2
        pars.set_accuracy(HighAccuracyDefault=True, AccuracyBoost=3.0)
        pars.set_cosmology(H0 = 100 * self.hubble, ombh2 = ombh2, omch2 = omch2, omk=0., mnu=self.m_nu, neutrino_hierarchy=self.nu_hierarchy, num_massive_neutrinos = 3)
        #Initial cosmology
        pars.InitPower.set_params(ns=self.ns, As=self.scalar_amp, pivot_scalar=2e-3, pivot_tensor=2e-3)
        #At which redshifts should we produce CAMB output: we want the starting redshift of the simulation,
        #but we also want some other values for checking purposes
        #Extra redshifts at which to generate CAMB output, in addition to self.redshift and self.redshift/2
        code = self.code_class_name(outdir=self.outdir, box=self.box, npart=self.npart, redshift=self.redshift, separate_gas=self.separate_gas, omega0=self.omega0, omegab=self.omegab, hubble=self.hubble, m_nu=self.m_nu, **self.code_args)
        camb_zz = np.concatenate([[self.redshift,], 1/code.generate_times()-1,[code.redend,]])
        pars.set_matter_power(redshifts = camb_zz, kmax = 2*math.pi*10*self.npart/self.box)
        pars.NonLinear = model.NonLinear_none
        #Get results
        results = camb.get_results(pars)

        transfers = results.get_matter_transfer_data()
        kh, camb_zz, pk = results.get_linear_matter_power_spectrum(have_power_spectra=True)
        cambpars = os.path.join(self.outdir, "_camb_params.ini")
        cfd = open(cambpars, 'w')
        #Write used parameters to a file
        print(pars, file=cfd)
        camb_output = "camb_linear/"
        camb_outdir = os.path.join(self.outdir,camb_output)
        try:
            os.mkdir(camb_outdir)
        except FileExistsError:
            pass
        #Set values: note we will write to camb_linear/ics_matterpow_99.dat with the below.
        for i, zz in enumerate(camb_zz):
            mfn = os.path.join(camb_outdir, "ics_matterpow_"+self._camb_zstr(zz)+".dat")
            #Get the power spectra
            matpow = np.vstack([kh, pk[i]])
            np.savetxt(mfn, matpow.T)
            tfn = os.path.join(camb_outdir,"ics_transfer_"+self._camb_zstr(zz)+".dat")
            np.savetxt(tfn, transfers.transfer_data[:,:,i].T)
        return camb_outdir

    def _camb_zstr(self,zz):
        """Get the formatted redshift for CAMB output files."""
        if zz > 10:
            zstr = str(int(zz))
        else:
            zstr = '%.1g' % zz
        return zstr

    def genicfile(self, camb_output):
        """Generate the GenIC parameter file"""
        config = configobj.ConfigObj(self.genicdefault)
        config.filename = os.path.join(self.outdir, self.genicout)
        config['Box'] = self.box*1000
        config['Nmesh'] = self.npart * 2
        genicout = "ICS"
        try:
            os.mkdir(os.path.join(self.outdir, genicout))
        except FileExistsError:
            pass
        config['OutputDir'] = genicout
        #Is this enough information, or should I add a short hash?
        genicfile = str(self.box)+"_"+str(self.npart)+"_"+str(self.redshift)
        config['FileBase'] = genicfile
        config['NCDM'] = self.npart
        config['NNeutrino'] = 0
        config['ICFormat'] = self.icformat
        if self.separate_gas:
            config['NBaryon'] = self.npart
            #The 2LPT correction is computed for one fluid. It is not clear
            #what to do with a second particle species, so turn it off.
            #Even for CDM alone there are corrections from radiation:
            #order: Omega_r / omega_m ~ 3 z/100 and it is likely
            #that the baryon 2LPT term is dominated by the CDM potential
            #(YAH, private communication)
            config['TWOLPT'] = 0
        else:
            config['NBaryon'] = 0
        #Total matter density, not CDM matter density.
        config['Omega'] = self.omega0
        config['OmegaLambda'] = 1- self.omega0
        config['OmegaBaryon'] = self.omegab
        config['HubbleParam'] = self.hubble
        config['Redshift'] = self.redshift
        zstr = self._camb_zstr(self.redshift)
        config['FileWithInputSpectrum'] = camb_output + "ics_matterpow_"+zstr+".dat"
        config['FileWithTransfer'] = camb_output + "ics_transfer_"+zstr+".dat"
        config['NumFiles'] = int(self.numfiles)
        assert config['InputSpectrum_UnitLength_in_cm'] == '3.085678e24'
        config['Seed'] = self.seed
        config['NU_Vtherm_On'] = 0
        config['NNeutrino'] = 0
        config['RayleighScatter'] = int(self.rscatter)
        config = self._genicfile_child_options(config)
        config.write()
        return (os.path.join(genicout, genicfile), config.filename)

    def _alter_power(self, camb_output):
        """Function to hook if you want to change the CAMB output power spectrum.
        Should save the new power spectrum to camb_output + _matterpow_str(redshift).dat"""
        zstr = self._camb_zstr(self.redshift)
        camb_file = os.path.join(camb_output,"ics_matterpow_"+zstr+".dat")
        os.stat(camb_file)
        return

    def _genicfile_child_options(self, config):
        """Set extra parameters in child classes"""
        return config

    def _fromarray(self):
        """Convert the data stored as lists back to what it was."""
        for arr in self._really_arrays:
            self.__dict__[arr] = np.array(self.__dict__[arr])
        self._really_arrays = []
        for arr in self._really_types:
            #Some crazy nonsense to convert the module, name
            #string tuple we stored back into a python type.
            mod = importlib.import_module(self.__dict__[arr][0])
            self.__dict__[arr] = getattr(mod, self.__dict__[arr][1])
        self._really_types = []

    def txt_description(self):
        """Generate a text file describing the parameters of the code that generated this simulation, for reproducibility."""
        #But ditch the output of make
        self.make_output = ""
        self._really_arrays = []
        self._really_types = []
        for nn, val in self.__dict__.items():
            #Convert arrays to lists
            if isinstance(val, np.ndarray):
                self.__dict__[nn] = val.tolist()
                self._really_arrays.append(nn)
            #Convert types to string tuples
            if isinstance(val, type):
                self.__dict__[nn] = (val.__module__, val.__name__)
                self._really_types.append(nn)
        with open(os.path.join(self.outdir, "SimulationICs.json"), 'w') as jsout:
            json.dump(self.__dict__,jsout)
        #Turn the changed types back.
        self._fromarray()

    def load_txt_description(self):
        """Load the text file describing the parameters of the code that generated a simulation."""
        with open(os.path.join(self.outdir, "SimulationICs.json"), 'r') as jsin:
            self.__dict__ = json.load(jsin)
        self._fromarray()

    def check_ic_power_spectra(self, camb_output, genicfileout,accuracy=0.05):
        """Generate the power spectrum for each particle type from the generated simulation files, using GenPK,
        and check that it matches the input. This is a consistency test on each simulation output."""
        #Generate power spectra
        output = os.path.join(self.outdir, genicfileout)
        #Now check that they match what we put into the simulation, from CAMB
        #Reload the CAMB files from disc, just in case something went wrong writing them.
        zstr = self._camb_zstr(self.redshift)
        matterpow = os.path.join(camb_output, "ics_matterpow_"+zstr+".dat")
        transfer = os.path.join(camb_output, "ics_transfer_"+zstr+".dat")
        cambpow = cambpower.CAMBPowerSpectrum(matterpow, transfer, kmin=2*math.pi/self.box/5, kmax = self.npart*2*math.pi/self.box*10)
        #Error to tolerate on simulated power spectrum
        #Check whether we output neutrinos
        for sp in ["DM","by"]:
            #GenPK output is at PK-[nu,by,DM]-basename(genicfileout)
            tt = '1/'
            if sp == "by":
                tt = '0/'
                if not self.separate_gas:
                    continue
            cat = BigFileCatalog(output, dataset=tt, header='Header')
            mesh = cat.to_mesh(Nmesh=self.npart*2, window='cic', compensated=True, interlaced=True)
            pk = FFTPower(cat, mode='1d', Nmesh=self.npart*2)
            #GenPK output is at PK-[nu,by,DM]-basename(genicfileout)
            #Load the power spectra
            #Convert units from kpc/h to Mpc/h
            kk_ic = pk.power['k'][1:]*1e3
            Pk_ic = pk.power['power'][1:].real/1e9
            #Load the power spectrum. Note that DM may incorporate other particle types.
            if not self.separate_gas and not self.separate_nu and sp =="DM":
                Pk_camb = cambpow.get_camb_power(kk_ic, species="tot")
            elif not self.separate_gas and self.separate_nu and sp == "DM":
                Pk_camb = cambpow.get_camb_power(kk_ic, species="DMby")
            #Case with self.separate_gas true and separate_nu false is assumed to have omega_nu = 0.
            else:
                Pk_camb = cambpow.get_camb_power(kk_ic, species=sp)
            #Check that they agree between 1/4 the box and 1/4 the nyquist frequency
            imax = np.searchsorted(kk_ic, self.npart*2*math.pi/self.box/4)
            imin = np.searchsorted(kk_ic, 2*math.pi/self.box*4)
            #Make some useful figures
            plt.semilogx(kk_ic, Pk_ic/Pk_camb,linewidth=2)
            plt.semilogx([kk_ic[0]*0.9,kk_ic[-1]*1.1], [0.95,0.95], ls="--",linewidth=2)
            plt.semilogx([kk_ic[0]*0.9,kk_ic[-1]*1.1], [1.05,1.05], ls="--",linewidth=2)
            plt.semilogx([kk_ic[imin],kk_ic[imin]], [0,1.5], ls=":",linewidth=2)
            plt.semilogx([kk_ic[imax],kk_ic[imax]], [0,1.5], ls=":",linewidth=2)
            plt.ylim(0., 1.5)
            plt.savefig("PK-IC-"+sp+"-diff.pdf")
            plt.clf()
            plt.loglog(kk_ic, Pk_ic,linewidth=2)
            plt.loglog(kk_ic, Pk_camb,ls="--", linewidth=2)
            plt.ylim(ymax=Pk_camb[0]*10)
            plt.savefig("PK-IC-"+sp+"-abs.pdf")
            plt.clf()
            error = abs(Pk_ic[imin:imax]/Pk_camb[imin:imax] -1)
            #Don't worry too much about one failing mode.
            if np.size(np.where(error > accuracy)) > 3:
                raise RuntimeError("Pk accuracy check failed for "+sp+". Max error: "+str(np.max(error)))

    def make_simulation(self, pkaccuracy=0.05, do_build=False):
        """Wrapper function to make the simulation ICs."""
        #First generate the input files for CAMB
        camb_output = self.cambfile()
        #Then run CAMB
        self.camb_git = camb.__version__
        #Change the power spectrum file on disc if we want to do that
        self._alter_power(os.path.join(self.outdir,camb_output))
        #Now generate the GenIC parameters
        (genic_output, genic_param) = self.genicfile(camb_output)
        #Run N-GenIC
        genic = utils.find_exec(self.genicexe)
        self.genic_git = utils.get_git_hash(genic)
        subprocess.check_call([genic, genic_param],cwd=self.outdir)
        #Save a json of ourselves.
        self.txt_description()
        #Check that the ICs have the right power spectrum
        self.check_ic_power_spectra(os.path.join(self.outdir,camb_output), genic_output,accuracy=pkaccuracy)
        #Make the parameter files.
        ics = self.code_class_name(outdir=self.outdir, box=self.box, npart=self.npart, redshift=self.redshift, separate_gas=self.separate_gas, omega0=self.omega0, omegab=self.omegab, hubble=self.hubble, m_nu=self.m_nu, **self.code_args)
        return ics.make_simulation(genic_output, do_build=do_build)
