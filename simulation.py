"""
Module to automate the generation of simulation config files.
The base class is Simulation, which creates the config files for a single simulation.
It is meant to be called from other classes as part of a suite,
More specialised simulation types can inherit from it.
Different machines can be implemented as decorators.
"""
from __future__ import print_function
import os
import os.path
import re
import math
import shutil
import glob
import subprocess
import json
import configobj
import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
from . import cambpower
from . import read_uvb_tab

def find_exec(executable):
    """Simple function to locate a binary in a nearby directory"""
    possible = [executable,]+glob.glob("depends/*/"+executable)+ glob.glob(os.path.join("../*/", executable))
    exists = [ex for ex in possible if os.path.exists(ex) and os.path.isfile(ex) ]
    if len(exists) > 1:
        print("Warning: found multiple possibilities: ",exists)
    if len(exists) > 0:
        return exists[0]
    raise ValueError(executable+" not found")

def get_git_hash(path):
    """Get the git hash of a file."""
    rpath = os.path.realpath(path)
    if not os.path.isdir(rpath):
        rpath = os.path.dirname(rpath)
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd = rpath, universal_newlines=True)
    return commit_hash

class Simulation(object):
    """
    Class for creating config files needed to run a single simulation.
    There are a few things this class needs to do:

    - Generate CAMB input files
    - Generate N-GenIC input files (to use CAMB output)
    - Run CAMB and N-GenIC to generate ICs
    - Generate Gadget input files that match the ICs

    The class will store the parameters of the simulation, and each public method will do one of these things.
    Many things are left hard-coded.
    We assume flatness.

    Init parameters:
    outdir - Directory in which to save ICs
    box - Box size in comoving Mpc/h
    npart - Cube root of number of particles
    separate_gas - if true the ICs will contain baryonic particles. If false, just DM.
    redshift - redshift at which to generate ICs
    omegab - baryon density. Note that if we do not have gas particles, still set omegab, but set separate_gas = False
    omegam - Matter density
    hubble - Hubble parameter, h, which is H0 / (100 km/s/Mpc)
    scalar_amp - Initial amplitude of scalar power spectrum to feed to CAMB
    ns - tilt of scalar power spectrum to feed to CAMB
    """
    def __init__(self, outdir, box, npart, *, seed = 9281110, redshift=99, redend = 0, separate_gas=True, separate_nu=False, omegac=0.2408, omegab=0.0472, omeganu=0.,hubble=0.7, scalar_amp=2.427e-9, ns=0.97, uvb="hm", do_build=True):
        #Check that input is reasonable and set parameters
        #In Mpc/h
        assert box < 20000
        self.box = box
        #Cube root
        assert npart > 1 and npart < 16000
        self.npart = int(npart)
        #Physically reasonable
        assert omegac <= 1 and omegac > 0
        self.omegac = omegac
        assert omegab > 0 and omegab < 1
        self.omegab = omegab
        assert omeganu >=0 and omeganu < omegac
        self.omeganu = omeganu
        assert redshift > 1 and redshift < 1100
        self.redshift = redshift
        assert redend >= 0 and redend < 1100
        self.redend = redend
        assert hubble < 1 and hubble > 0
        self.hubble = hubble
        assert scalar_amp < 1e-8 and scalar_amp > 0
        self.scalar_amp = scalar_amp
        assert ns > 0 and ns < 2
        self.ns = ns
        #Structure seed.
        self.seed = seed
        #Baryons?
        self.separate_gas = separate_gas
        #Neutrinos?
        self.separate_nu = separate_nu
        #UVB? Only matters if gas
        self.uvb = uvb
        assert self.uvb == "hm" or self.uvb == "fg"
        #CPU parameters: these are specified to a default here, but should be over-ridden in a machine-specific decorator.
        self.nproc = 8
        self.email = "sbird4@jhu.edu"
        self.timelimit = 10
        #Maximum memory available for an MPI task
        self.memory = 1800
        #Will we try to build gadget?
        self.do_build = do_build
        #Number of files per snapshot
        #This is chosen to give a reasonable number and
        #a constant number of particles per file.
        self.numfiles = int(np.max([2,self.npart**3//2**24]))
        #Maximum number of files to write in parallel.
        #Cannot be larger than number of processors
        self.maxpwrite = self.nproc
        #Total matter density
        self.omega0 = self.omegac + self.omegab + self.omeganu
        outdir = os.path.realpath(os.path.expanduser(outdir))
        #Make the output directory: will fail if parent does not exist
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        else:
            if os.listdir(outdir) != []:
                print("Warning: ",outdir," is non-empty")
        self.outdir = outdir
        defaultpath = os.path.dirname(__file__)
        #Default values for the CAMB parameters
        self.cambdefault = os.path.join(defaultpath,"params.ini")
        #Filename for new CAMB file
        self.cambout = "_camb_params.ini"
        #Default GenIC paths
        self.genicdefault = os.path.join(defaultpath,"ngenic.param")
        self.genicout = "_genic_params.ini"
        #Default parameter file names
        self.gadgetdefaultparam = os.path.join(defaultpath,"gadgetparams.param")
        self.gadgetparam = "gadget3.param"
        #Executable names
        self.cambexe = "camb"
        self.gadgetexe = "P-Gadget3"
        self.gadgetconfig = "Config.sh"
        self.gadget_dir = os.path.expanduser("~/codes/P-Gadget3/")
        self.genicexe = "N-GenIC"
        #Output times
        #Extra redshifts at which to generate CAMB output, in addition to self.redshift and self.redshift/2
        self.camb_times = [9,4,2,1,0]
        #For repeatability, we store git hashes of Gadget, GenIC, CAMB and ourselves
        #at time of running.
        self.simulation_git = get_git_hash(".")

    def cambfile(self):
        """Generate the CAMB parameter file from the (cosmological) simulation parameters and the default values"""
        #Load CAMB file using ConfigObj
        config = configobj.ConfigObj(self.cambdefault)
        config.filename = os.path.join(self.outdir, self.cambout)
        #Set values
        camb_outdir = os.path.join(self.outdir,"camb_linear")
        try:
            os.mkdir(camb_outdir)
        except FileExistsError:
            pass
        camb_output = camb_outdir+"/ics"
        config['output_root'] = camb_output
        #Can't change this easily because the parameters then have different names
        assert config['use_physical'] == 'T'
        config['hubble'] = self.hubble * 100
        config['ombh2'] = self.omegab*self.hubble**2
        config['omch2'] = self.omegac*self.hubble**2
        config['omnuh2'] = self.omeganu*self.hubble**2
        config['omk'] = 0.
        #Initial power spectrum: MAKE SURE you set the pivot scale to the WMAP value!
        config['pivot_scalar'] = 2e-3
        config['pivot_tensor'] = 2e-3
        config['scalar_spectral_index(1)'] = self.ns
        config['scalar_amp(1)'] = self.scalar_amp
        #Various numerical parameters
        #Maximum relevant scale is 2 pi * avg. interparticle spacing * 2. Set kmax to double this.
        config['transfer_kmax'] = 2*math.pi*4*self.npart/self.box
        #At which redshifts should we produce CAMB output: we want the starting redshift of the simulation,
        #but we also want some other values for checking purposes
        redshifts = [self.redshift, (self.redshift+1)/2-1] + self.camb_times
        for (n,zz) in zip(range(1,len(redshifts)+1), redshifts):
            config['transfer_redshift('+str(n)+')'] = zz
            config['transfer_filename('+str(n)+')'] = 'transfer_'+str(zz)+'.dat'
            config['transfer_matterpower('+str(n)+')'] = 'matterpow_'+str(zz)+'.dat'
        config['transfer_num_redshifts'] = len(redshifts)
        #Set up the neutrinos.
        #This has it's own function so it can be overriden by child classes
        config = self._camb_neutrinos(config)
        #Write the config file
        config.write()
        return (camb_output, config.filename)

    def _camb_neutrinos(self, config):
        """Modify the CAMB config file to have massless neutrinos.
        Designed to be easily over-ridden"""
        config['massless_neutrinos'] = 3.046
        config['massive_neutrinos'] = 0
        return config

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
        config['OutputDir'] = os.path.join(self.outdir, genicout)
        #Is this enough information, or should I add a short hash?
        genicfile = str(self.box)+"_"+str(self.npart)+"_"+str(self.redshift)
        config['FileBase'] = genicfile
        config['NCDM'] = self.npart
        if self.separate_gas:
            config['NBaryon'] = self.npart
            #The 2LPT correction is computed for one fluid. It is not clear
            #what to do with a second particle species, so turn it off.
            #Even for CDM alone there are corrections from radiation:
            #order: Omega_r / omega_m ~ 3 z/100 and it is likely
            #that the baryon 2LPT term is dominated by the CDM potential
            #(YAH, private communication)
            config['TWOLPT'] = 0
        #Total matter density, not CDM matter density.
        config['Omega'] = self.omega0
        config['OmegaLambda'] = 1- self.omega0
        config['OmegaBaryon'] = self.omegab
        config['OmegaDM_2ndSpecies'] = self.omeganu
        config['HubbleParam'] = self.hubble
        config['Redshift'] = self.redshift
        config['FileWithInputSpectrum'] = camb_output + "_matterpow_"+str(self.redshift)+".dat"
        config['FileWithTransfer'] = camb_output + "_transfer_"+str(self.redshift)+".dat"
        config['NumFiles'] = int(self.numfiles)
        assert config['InputSpectrum_UnitLength_in_cm'] == '3.085678e24'
        config['Seed'] = self.seed
        config['NU_On'] = 0
        config['NNeutrino'] = 0
        config = self._genicfile_child_options(config)
        config.write()
        return (os.path.join(genicout, genicfile), config.filename)

    def _genicfile_child_options(self, config):
        """Set extra parameters in child classes"""
        return config

    def gadget3config(self, prefix=""):
        """Generate a Gadget Config.sh file. This doesn't fit nicely into configobj.
        Many of the simulation parameters are stored here, but none of the cosmology.
        Some of these parameters are cluster dependent.
        We are assuming Gadget-3. Arepo or Gadget-2 need a different set of options."""
        g_config_filename = os.path.join(self.outdir, self.gadgetconfig)
        with open(g_config_filename,'w') as config:
            config.write(prefix+"PERIODIC\n")
            #Can be reduced for lower memory but lower speed.
            config.write(prefix+"PMGRID="+str(self.npart*2)+"\n")
            #These are memory options: if short on memory, change them.
            #MULTIPLEDOMAINS speeds up somewhat
            config.write(prefix+"MULTIPLEDOMAINS=4\n")
            config.write(prefix+"TOPNODEFACTOR=3.0\n")
            #Again, can be turned off for lower memory usage
            #but changes output format
            config.write(prefix+"LONGIDS\n")
            config.write(prefix+"PEANOHILBERT\n")
            config.write(prefix+"WALLCLOCK\n")
            config.write(prefix+"MYSORT\n")
            config.write(prefix+"MOREPARAMS\n")
            config.write(prefix+"POWERSPEC_ON_OUTPUT\n")
            config.write(prefix+"POWERSPEC_ON_OUTPUT_EACH_TYPE\n")
            #Changes H(z)
            config.write(prefix+"INCLUDE_RADIATION\n")
            config.write(prefix+"HAVE_HDF5\n")
            #We may need this sometimes, depending on the machine
            #Options for gas simulations
            self._cluster_config_options(config, prefix)
            if self.separate_gas:
                config.write(prefix+"COOLING\n")
                #This needs implementing
                #config.write(prefix+"UVB_SELF_SHIELDING")
                #Optional feedback model options
                self._feedback_config_options(config, prefix)
            self._gadget3_child_options(config, prefix)
        return g_config_filename

    def _cluster_config_options(self,config, prefix=""):
        """Config options that might be specific to a particular cluster"""
        _ = (config, prefix)
        #isend/irecv is quite slow on some clusters because of the extra memory allocations.
        #Maybe test this on your specific system and see if it helps.
        #config.write(prefix+"NO_ISEND_IRECV_IN_DOMAIN\n")
        #config.write(prefix+"NO_ISEND_IRECV_IN_PM\n")
        #config.write(prefix+"NOTYPEPREFIX_FFTW\n")
        return

    def _feedback_config_options(self, config, prefix=""):
        """Options in the Config.sh file for a potential star-formation/feedback model"""
        config.write(prefix+"SFR")
        return

    def _gadget3_child_options(self, _, __):
        """Gadget-3 compilation options for Config.sh which should be written by the child class."""
        return

    def gadget3params(self, genicfileout):
        """Gadget 3 parameter file. Almost a configobj, but needs a regex at the end to change # to % and remove '='.
        Again, will be different for Arepo and Gadget2.
        Arguments:
            genicfileout - where the ICs are saved
            timelimit - simulation time limit in hours"""
        config = configobj.ConfigObj(self.gadgetdefaultparam)
        config.filename = os.path.join(self.outdir, self.gadgetparam)
        config['InitCondFile'] = genicfileout
        config['OutputDir'] = "output"
        try:
            os.mkdir(os.path.join(self.outdir, "output"))
        except FileExistsError:
            pass
        config['SnapshotFileBase'] = "snap"
        config['TimeLimitCPU'] = int(60*60*self.timelimit*20/17.-3000)
        config['TimeBegin'] = 1./(1+self.redshift)
        config['TimeMax'] = 1./(1+self.redend)
        config['Omega0'] = self.omega0
        config['OmegaLambda'] = 1- self.omega0
        #OmegaBaryon should be zero for gadget if we don't have gas particles
        config['OmegaBaryon'] = self.omegab*self.separate_gas
        config['HubbleParam'] = self.hubble
        config['BoxSize'] = self.box * 1000
        config['OutputListOn'] = 1
        timefile = "times.txt"
        config['OutputListFilename'] = timefile
        self._print_times(timefile)
        #This should just be larger than the simulation time limit
        config['CpuTimeBetRestartFile'] = 60*60*self.timelimit*10
        config['NumFilesPerSnapshot'] = self.numfiles
        #There is a maximum here because some filesystems may not like parallel writes!
        config['NumFilesWrittenInParallel'] = np.min([self.maxpwrite, self.numfiles])
        #Softening is 1/30 of the mean linear interparticle spacing
        soften = 1000 * self.box/self.npart/30.
        for ptype in ('Gas', 'Halo', 'Disk', 'Bulge', 'Stars', 'Bndry'):
            config['Softening'+ptype] = soften
            config['Softening'+ptype+'MaxPhys'] = soften
        config['ICFormat'] = 3
        config['SnapFormat'] = 3
        config['RestartFile'] = "restart"
        #This could be tuned in lower memory conditions
        config['BufferSize'] = 100
        if self.separate_gas:
            config['CoolingOn'] = 1
            config = self._sfr_params(config)
            config = self._feedback_params(config)
            #Copy a TREECOOL file into the right place.
            self._copy_uvb()
            #Need more memory for a feedback model
            config['PartAllocFactor'] = 4
        else:
            config['PartAllocFactor'] = 2
        config['MaxMemSize'] = self.memory
        #Add other config parameters
        config = self._other_params(config)
        config.write()
        #Now we need to regex the generated file to fit the gadget format
        #This is somewhat unsafe, but who cares?
        cf = open(config.filename,'r')
        configstr = cf.read()
        configstr = re.sub("#","%",configstr)
        configstr = re.sub("="," ",configstr)
        cf.close()
        cf = open(config.filename,'w')
        cf.write(configstr)
        cf.close()
        return

    def _sfr_params(self, config):
        """Config parameters for the default Springel & Hernquist star formation model"""
        config['StarformationOn'] = 1
        config['CritPhysDensity'] =  0
        config['MaxSfrTimescale'] = 1.5
        config['CritOverDensity'] = 1000.0
        config['TempSupernova'] = 1e+08
        config['TempClouds'] = 1000
        config['FactorSN'] = 0.1
        config['FactorEVP'] = 1000
        return config

    def _feedback_params(self, config):
        """Config parameters for the feedback models"""
        return config

    def _other_params(self, config):
        """Function to override to set other config parameters"""
        return config

    def _alter_power(self, camb_output):
        """Function to hook if you want to change the CAMB output power spectrum.
        Should save the new power spectrum to camb_output + _matterpow_str(redshift).dat"""
        camb_file = camb_output+"_matterpow_"+str(self.redshift)+".dat"
        os.stat(camb_file)
        return

    def _generate_times(self):
        """List of output times for a simulation. Can be overridden,
        but default is evenly spaced in a from start to end."""
        astart = 1./(1+self.redshift)
        aend = 1./(1+self.redend)
        times = np.linspace(astart, aend,9)
        return times

    def _copy_uvb(self):
        """The UVB amplitude for Gadget is specified in a file named TREECOOL in the same directory as the gadget binary."""
        fuvb = read_uvb_tab.get_uvb_filename(self.uvb)
        shutil.copy(fuvb, os.path.join(self.outdir,"TREECOOL"))

    def _print_times(self, timefile):
        """Print times to the times.txt file"""
        times = self._generate_times()
        np.savetxt(os.path.join(self.outdir, timefile), times)

    def generate_mpi_submit(self):
        """Generate a sample mpi_submit file.
        The prefix argument is a string at the start of each line.
        It separates queueing system directives from normal comments"""
        with open(os.path.join(self.outdir, "mpi_submit"),'w') as mpis:
            mpis.write("#!/bin/bash\n")
            mpis.write(self._queue_directive())
            mpis.write(self._mpi_program())

    def _mpi_program(self):
        """String for MPI program to execute"""
        qstring = "mpirun -np "+str(self.nproc)+" "+self.gadgetexe+" "+self.gadgetparam+"\n"
        return qstring

    def _queue_directive(self, prefix="#PBS"):
        """Write the part of the mpi_submit file that directs the queueing system.
        This is usually specific to a given cluster.
        The prefix argument is a string at the start of each line.
        It separates queueing system directives from normal comments"""
        qstring = prefix+" -j eo\n"
        qstring += prefix+" -m bae\n"
        qstring += prefix+" -M "+self.email+"\n"
        qstring += prefix+" -l walltime="+str(self.timelimit)+":00:00\n"
        return qstring

    def txt_description(self):
        """Generate a text file describing the parameters of the code that generated this simulation, for reproducibility."""
        #But ditch the output of make
        self.make_output = ""
        with open(os.path.join(self.outdir, "Simulation.json"), 'w') as jsout:
            json.dump(self.__dict__, jsout)

    def load_txt_description(self):
        """Load the text file describing the parameters of the code that generated a simulation."""
        with open(os.path.join(self.outdir, "Simulation.json"), 'r') as jsin:
            self.__dict__ = json.load(jsin)

    def make_simulation(self):
        """Wrapper function to make all the simulation parameter files in turn and run the binaries"""
        #First generate the input files for CAMB
        (camb_output, camb_param) = self.cambfile()
        #Then run CAMB
        camb = find_exec(self.cambexe)
        self.camb_git = get_git_hash(camb)
        #In python 3.5, can use subprocess.run to do this.
        #But for backwards compat, use check_output
        subprocess.check_call([os.path.join(os.getcwd(),camb), camb_param], cwd=os.path.dirname(camb))
        #Change the power spectrum file on disc if we want to do that
        self._alter_power(camb_output)
        #Now generate the GenIC parameters
        (genic_output, genic_param) = self.genicfile(camb_output)
        #Run N-GenIC
        genic = find_exec(self.genicexe)
        self.genic_git = get_git_hash(genic)
        subprocess.check_call([genic, genic_param])
        #Generate Gadget makefile
        gadget_config = self.gadget3config()
        #Symlink the new gadget config to the source directory
        if self.do_build:
            self.do_gadget_build(gadget_config)
        #Generate Gadget parameter file
        self.gadget3params(genic_output)
        #Generate mpi_submit file
        self.generate_mpi_submit()
        #Save a json of ourselves.
        self.txt_description()
        #Check that the ICs have the right power spectrum
        self.check_ic_power_spectra(camb_output, genic_output)

    def do_gadget_build(self, gadget_config):
        """Make a gadget build and check it succeeded."""
        os.remove(os.path.join(self.gadget_dir, self.gadgetconfig))
        os.symlink(gadget_config, os.path.join(self.gadget_dir, self.gadgetconfig))
        #Build gadget
        gadget_binary = os.path.join(self.gadget_dir, self.gadgetexe)
        try:
            g_mtime = os.stat(gadget_binary).st_mtime
        except FileNotFoundError:
            g_mtime = -1
        self.gadget_git = get_git_hash(gadget_binary)
        self.make_output = subprocess.check_output(["make", "-j8"], cwd=self.gadget_dir, universal_newlines=True)
        #Check that the last-changed time of the binary has actually changed..
        assert g_mtime != os.stat(gadget_binary).st_mtime
        #Copy the gadget binary to the new location
        shutil.copy(os.path.join(self.gadget_dir, self.gadgetexe), os.path.join(self.outdir,self.gadgetexe))

    def check_ic_power_spectra(self, camb_output, genicfileout):
        """Generate the power spectrum for each particle type from the generated simulation files, using GenPK,
        and check that it matches the input. This is a consistency test on each simulation output."""
        #Generate power spectra
        genpk = find_exec("gen-pk")
        genicfileout = os.path.join(self.outdir, genicfileout)
        subprocess.check_call([genpk, "-i", genicfileout, "-o", os.path.dirname(genicfileout)])
        #Now check that they match what we put into the simulation, from CAMB
        #Reload the CAMB files from disc, just in case something went wrong writing them.
        matterpow = camb_output + "_matterpow_"+str(self.redshift)+".dat"
        transfer = camb_output + "_transfer_"+str(self.redshift)+".dat"
        camb = cambpower.CAMBPowerSpectrum(matterpow, transfer, kmin=2*math.pi/self.box/5, kmax = self.npart*2*math.pi/self.box*10)
        #Error to tolerate on simulated power spectrum
        accuracy = 0.05
        for sp in ["DM","by", "nu"]:
            #GenPK output is at PK-[nu,by,DM]-basename(genicfileout)
            gpkout = "PK-"+sp+"-"+os.path.basename(genicfileout)
            go = os.path.join(os.path.dirname(genicfileout), gpkout)
            if sp == "DM" or (self.separate_gas and sp == "by"):
                assert os.path.exists(go)
            elif not os.path.exists(go):
                continue
            #Load the power spectra
            (kk_ic, Pk_ic) = load_genpk(go, self.box)
            #Load the power spectrum. Note that DM may incorporate other particle types.
            if not self.separate_gas and not self.separate_nu and sp =="DM":
                Pk_camb = camb.get_camb_power(kk_ic, species="tot")
            elif not self.separate_gas and self.separate_nu and sp == "DM":
                Pk_camb = camb.get_camb_power(kk_ic, species="DMby")
            #Case with self.separate_gas true and separate_nu false is assumed to have omega_nu = 0.
            else:
                Pk_camb = camb.get_camb_power(kk_ic, species=sp)
            #Check that they agree between 1/4 the box and 1/4 the nyquist frequency
            imax = np.searchsorted(kk_ic, self.npart*2*math.pi/self.box/4)
            imin = np.searchsorted(kk_ic, 2*math.pi/self.box*4)
            if sp == "nu":
                #Neutrinos get special treatment here.
                #Because they don't really cluster, getting the initial power really right
                #(especially on small scales) is both hard and rather futile.
                accuracy = 0.1
                ii = np.where(Pk_ic < Pk_ic[0]*1e-5)
                if np.size(ii) > 0:
                    imax = ii[0][0]
            #Make some useful figures
            plt.semilogx(kk_ic, Pk_ic/Pk_camb,linewidth=2)
            plt.semilogx([kk_ic[0]*0.9,kk_ic[-1]*1.1], [0.95,0.95], ls="--",linewidth=2)
            plt.semilogx([kk_ic[0]*0.9,kk_ic[-1]*1.1], [1.05,1.05], ls="--",linewidth=2)
            plt.semilogx([kk_ic[imin],kk_ic[imin]], [0,1.5], ls=":",linewidth=2)
            plt.semilogx([kk_ic[imax],kk_ic[imax]], [0,1.5], ls=":",linewidth=2)
            plt.ylim(0., 1.5)
            plt.savefig(go+"-diff.pdf")
            plt.clf()
            plt.loglog(kk_ic, Pk_ic,linewidth=2)
            plt.loglog(kk_ic, Pk_camb,ls="--", linewidth=2)
            plt.ylim(ymax=Pk_camb[0]*10)
            plt.savefig(go+"-abs.pdf")
            plt.clf()
            error = abs(Pk_ic[imin:imax]/Pk_camb[imin:imax] -1)
            #Don't worry too much about one failing mode.
            if np.size(np.where(error > accuracy)) > 3:
                raise RuntimeError("Pk accuracy check failed for "+sp+". Max error: "+str(np.max(error)))

def load_genpk(infile, box, minmode=1):
    """Load a power spectrum from a Gen-PK output, modifying units to agree with CAMB"""
    matpow = np.loadtxt(infile)
    scale = 2*math.pi/box
    kk = matpow[:,0]*scale
    Pk = matpow[:,1]/scale**3*(2*math.pi)**3
    count = matpow[:,2]
    #Rebin so that there are at least n modes per bin
    Pk_rebin = []
    kk_rebin = []
    lcount = 0
    istart = 0
    iend = 0
    while iend < np.size(kk):
        lcount+=count[iend]
        iend+=1
        if lcount >= minmode:
            p = np.sum(count[istart:iend]*Pk[istart:iend])/lcount
            assert p > 0
            k = np.sum(count[istart:iend]*kk[istart:iend])/lcount
            assert k > 0
            kk_rebin.append(k)
            Pk_rebin.append(p)
            istart=iend
            lcount=0
    return (np.array(kk_rebin), np.array(Pk_rebin))
