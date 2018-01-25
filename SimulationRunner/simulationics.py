"""Class to generate simulation ICS, separated out for clarity."""
from __future__ import print_function
import os.path
import math
import subprocess
import json
import shutil
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
from . import cambpower
from . import utils
from . import clusters
from . import read_uvb_tab

class SimulationICs(object):
    """
    Class for creating the initial conditions for a simulation.
    There are a few things this class needs to do:

    - Generate CAMB input files
    - Generate MP-GenIC input files (to use CAMB output)
    - Run CAMB and MP-GenIC to generate ICs

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
    def __init__(self, *, outdir, box, npart, seed = 9281110, redshift=99, redend=0, separate_gas=True, omega0=0.288, omegab=0.0472, hubble=0.7, scalar_amp=2.427e-9, ns=0.97, rscatter=False, m_nu=0, nu_hierarchy='degenerate', uvb="hm", cluster_class=clusters.MARCCClass):
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
        assert redend >= 0 and redend < 1100
        self.redend = redend
        assert hubble < 1 and hubble > 0
        self.hubble = hubble
        assert scalar_amp < 1e-7 and scalar_amp > 0
        self.scalar_amp = scalar_amp
        assert ns > 0 and ns < 2
        self.ns = ns
        #UVB? Only matters if gas
        self.uvb = uvb
        assert self.uvb == "hm" or self.uvb == "fg" or self.uvb == "sh"
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
        self._set_default_paths()
        self._cluster = cluster_class(gadget=self.gadgetexe, param=self.gadgetparam)
        #For repeatability, we store git hashes of Gadget, GenIC, CAMB and ourselves
        #at time of running.
        self.simulation_git = utils.get_git_hash(os.path.dirname(__file__))

    def _set_default_paths(self):
        """Default paths and parameter names."""
        #Default parameter file names
        self.gadgetparam = "mpgadget.param"
        self.genicout = "_genic_params.ini"
        #Executable names
        self.gadgetexe = "MP-Gadget"
        self.genicexe = "MP-GenIC"
        defaultpath = os.path.dirname(__file__)
        #Default GenIC paths
        self.genicdefault = os.path.join(defaultpath,"mpgenic.ini")
        self.gadgetconfig = "Options.mk"
        self.gadget_dir = os.path.expanduser("~/codes/MP-Gadget/")
        self.gadget_binary_dir = os.path.join(self.gadget_dir,"build")

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
        camb_zz = np.concatenate([[self.redshift,], 1/self.generate_times()-1,[self.redend,]])
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
        config['BoxSize'] = self.box*1000
        genicout = "ICS"
        try:
            os.mkdir(os.path.join(self.outdir, genicout))
        except FileExistsError:
            pass
        config['OutputDir'] = genicout
        #Is this enough information, or should I add a short hash?
        genicfile = str(self.box)+"_"+str(self.npart)+"_"+str(self.redshift)
        config['FileBase'] = genicfile
        config['Ngrid'] = self.npart
        config['NgridNu'] = 0
        config['ProduceGas'] = int(self.separate_gas)
        #The 2LPT correction is computed for one fluid. It is not clear
        #what to do with a second particle species, so turn it off.
        #Even for CDM alone there are corrections from radiation:
        #order: Omega_r / omega_m ~ 3 z/100 and it is likely
        #that the baryon 2LPT term is dominated by the CDM potential
        #(YAH, private communication)
        #Total matter density, not CDM matter density.
        config['Omega0'] = self.omega0
        config['OmegaLambda'] = 1- self.omega0
        config['OmegaBaryon'] = self.omegab
        config['HubbleParam'] = self.hubble
        config['Redshift'] = self.redshift
        zstr = self._camb_zstr(self.redshift)
        config['FileWithInputSpectrum'] = camb_output + "ics_matterpow_"+zstr+".dat"
        config['FileWithTransferFunction'] = camb_output + "ics_transfer_"+zstr+".dat"
        assert config['WhichSpectrum'] == '2'
        assert config['RadiationOn'] == '1'
        assert config['DifferentTransferFunctions'] == '1'
        assert config['InputPowerRedshift'] == '-1'
        assert config['InputSpectrum_UnitLength_in_cm'] == '3.085678e24'
        config['Seed'] = self.seed
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
        cc = self._cluster
        self._cluster = 0
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
        self._cluster = cc

    def load_txt_description(self):
        """Load the text file describing the parameters of the code that generated a simulation."""
        cc = self._cluster
        with open(os.path.join(self.outdir, "SimulationICs.json"), 'r') as jsin:
            self.__dict__ = json.load(jsin)
        self._fromarray()
        self._cluster = cc

    def gadget3config(self, prefix="OPT += -D"):
        """Generate a config Options file for Yu Feng's MP-Gadget.
        This code is intended tobe configured primarily via runtime options.
        Many of the Gadget options are always on, and there is a new PM gravity solver."""
        g_config_filename = os.path.join(self.outdir, self.gadgetconfig)
        with open(g_config_filename,'w') as config:
            config.write("# off-tree build into $(DESTDIR)\nDESTDIR = build\n")
            config.write("MPICC = mpicc\nMPICXX = mpic++\n")
            optimize = self._cluster.cluster_optimize()
            config.write("OPTIMIZE = "+optimize+"\n")
            config.write("GSL_INCL = $(shell gsl-config --cflags)\nGSL_LIBS = $(shell gsl-config --libs)\n")
            #We may want DENSITY_INDEPENDENT_SPH as well.
            #config.write(prefix+"DENSITY_INDEPENDENT_SPH\n")
            config.write(prefix+"OPENMP_USE_SPINLOCK\n")
            self._cluster.cluster_config_options(config, prefix)
            if self.separate_gas:
                #This needs implementing
                config.write(prefix+"SFR\n")
                #config.write(prefix+"UVB_SELF_SHIELDING")
                #Optional feedback model options
                self._feedback_config_options(config, prefix)
            self._gadget3_child_options(config, prefix)
        return g_config_filename

    def _feedback_config_options(self, config, prefix=""):
        """Options in the Config.sh file for a potential star-formation/feedback model"""
        config.write(prefix+"BLACK_HOLES\n")
        return

    def _gadget3_child_options(self, _, __):
        """Gadget-3 compilation options for Config.sh which should be written by the child class
        This is MP-Gadget, so it is likely there are none."""
        return

    def gadget3params(self, genicfileout):
        """MP-Gadget parameter file. This *is* a configobj.
        Note MP-Gadget supprts default arguments, so no need for a defaults file.
        Arguments:
            genicfileout - where the ICs are saved
        """
        config = configobj.ConfigObj()
        filename = os.path.join(self.outdir, self.gadgetparam)
        config.filename = filename
        config['InitCondFile'] = genicfileout
        config['OutputDir'] = "output"
        try:
            os.mkdir(os.path.join(self.outdir, "output"))
        except FileExistsError:
            pass
        config['TimeLimitCPU'] = int(60*60*self._cluster.timelimit*20/17.-3000)
        config['TimeMax'] = 1./(1+self.redend)
        config['Omega0'] = self.omega0
        config['OmegaLambda'] = 1- self.omega0
        #OmegaBaryon should be zero for gadget if we don't have gas particles
        config['OmegaBaryon'] = self.omegab*self.separate_gas
        config['HubbleParam'] = self.hubble
        config['RadiationOn'] = 1
        config['HydroOn'] = 1
        config['Nmesh'] = 2*self.npart
        config['SnapshotWithFOF'] = 1
        config['FOFHaloLinkingLength'] = 0.2
        config['OutputList'] =  ','.join([str(t) for t in self.generate_times()])
        #These are only used for gas, but must be set anyway
        config['MinGasTemp'] = 100
        #In equilibrium with the CMB at early times.
        config['InitGasTemp'] = 2.7*(1+self.redshift)
        #Set the required neutrino parameters.
        config['MassiveNuLinRespOn'] = 0
        config['LinearTransferFunction'] = "camb_linear/ics_transfer_"+str(self.redshift)+".dat"
        #This needs to be here until I fix the flux extractor to allow quintic kernels.
        config['DensityKernelType'] = 'cubic'
        config['PartAllocFactor'] = 2
        config['WindOn'] = 0
        config['WindModel'] = 'nowind'
        config['BlackHoleOn'] = 0
        if self.separate_gas:
            config['CoolingOn'] = 1
            config['TreeCoolFile'] = "TREECOOL"
            #Copy a TREECOOL file into the right place.
            self._copy_uvb()
            config = self._sfr_params(config)
            config = self._feedback_params(config)
        else:
            config['CoolingOn'] = 0
            config['StarformationOn'] = 0
        #Add other config parameters
        config = self._other_params(config)
        config.write()
        return

    def _sfr_params(self, config):
        """Config parameters for the default Springel & Hernquist star formation model"""
        config['StarformationOn'] = 1
        config['StarformationCriterion'] = 'density'
        return config

    def _feedback_params(self, config):
        """Config parameters for the feedback models"""
        return config

    def _other_params(self, config):
        """Function to override to set other config parameters"""
        return config

    def generate_times(self):
        """List of output times for a simulation. Can be overridden."""
        astart = 1./(1+self.redshift)
        aend = 1./(1+self.redend)
        times = np.array([0.02,0.1,0.2,0.25,0.3333,0.5,0.66667,0.83333])
        ii = np.where((times > astart)*(times < aend))
        assert np.size(times[ii]) > 0
        return times[ii]

    def _copy_uvb(self):
        """The UVB amplitude for Gadget is specified in a file named TREECOOL in the same directory as the gadget binary."""
        fuvb = read_uvb_tab.get_uvb_filename(self.uvb)
        shutil.copy(fuvb, os.path.join(self.outdir,"TREECOOL"))

    def _print_times(self, timefile):
        """Print times to the times.txt file"""
        times = self.generate_times()
        np.savetxt(os.path.join(self.outdir, timefile), times)

    def check_ic_power_spectra(self, camb_output, genicfileout,accuracy=0.05):
        """Generate the power spectrum for each particle type from the generated simulation files, using GenPK,
        and check that it matches the input. This is a consistency test on each simulation output."""
        #Generate power spectra
        output = os.path.join(self.outdir, genicfileout)
        #Now check that they match what we put into the simulation, from CAMB
        #Reload the CAMB files from disc, just in case something went wrong writing them.
        zstr = self._camb_zstr(self.redshift)
        matterpow = "camb_linear/ics_matterpow_"+zstr+".dat"
        transfer = "camb_linear/ics_transfer_"+zstr+".dat"
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
            cat.to_mesh(Nmesh=self.npart*2, window='cic', compensated=True, interlaced=True)
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
            plt.savefig(os.path.join(self.outdir,"ICS/PK-IC-"+sp+"-diff.pdf"))
            plt.clf()
            plt.loglog(kk_ic, Pk_ic,linewidth=2)
            plt.loglog(kk_ic, Pk_camb,ls="--", linewidth=2)
            plt.ylim(ymax=Pk_camb[0]*10)
            plt.savefig(os.path.join(self.outdir,"ICS/PK-IC-"+sp+"-abs.pdf"))
            plt.clf()
            error = abs(Pk_ic[imin:imax]/Pk_camb[imin:imax] -1)
            #Don't worry too much about one failing mode.
            if np.size(np.where(error > accuracy)) > 3:
                raise RuntimeError("Pk accuracy check failed for "+sp+". Max error: "+str(np.max(error)))

    def do_gadget_build(self, gadget_config):
        """Make a gadget build and check it succeeded."""
        conffile = os.path.join(self.gadget_dir, self.gadgetconfig)
        if os.path.islink(conffile):
            os.remove(conffile)
        if os.path.exists(conffile):
            os.rename(conffile, conffile+".backup")
        os.symlink(gadget_config, conffile)
        #Build gadget
        gadget_binary = os.path.join(self.gadget_binary_dir, self.gadgetexe)
        try:
            g_mtime = os.stat(gadget_binary).st_mtime
        except FileNotFoundError:
            g_mtime = -1
        self.gadget_git = utils.get_git_hash(gadget_binary)
        try:
            self.make_output = subprocess.check_output(["make", "-j"], cwd=self.gadget_dir, universal_newlines=True, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise
        #Check that the last-changed time of the binary has actually changed..
        assert g_mtime != os.stat(gadget_binary).st_mtime
        shutil.copy(gadget_binary, os.path.join(os.path.dirname(gadget_config),self.gadgetexe))

    def generate_mpi_submit(self):
        """Generate a sample mpi_submit file.
        The prefix argument is a string at the start of each line.
        It separates queueing system directives from normal comments"""
        self._cluster.generate_mpi_submit(self.outdir)

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
        subprocess.check_call([os.path.join(self.gadget_binary_dir,self.genicexe), genic_param],cwd=self.outdir)
        #Save a json of ourselves.
        self.txt_description()
        #Check that the ICs have the right power spectrum
        self.check_ic_power_spectra(os.path.join(self.outdir,camb_output), genic_output,accuracy=pkaccuracy)
        #Generate Gadget makefile
        gadget_config = self.gadget3config()
        #Symlink the new gadget config to the source directory
        if do_build:
            self.do_gadget_build(gadget_config)
        #Generate Gadget parameter file
        self.gadget3params(genic_output)
        #Generate mpi_submit file
        self.generate_mpi_submit()
        return gadget_config
