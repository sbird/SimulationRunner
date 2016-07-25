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
import shutil
import subprocess
import jsonpickle
import configobj
import numpy as np
from . import read_uvb_tab
from . import utils

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
    def __init__(self, *, outdir, box, npart, redshift=99, redend = 0, separate_gas=True, omegac=0.2408, omegab=0.0472, omeganu=0.,hubble=0.7, uvb="hm", do_build=True):
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
        #Baryons?
        self.separate_gas = separate_gas
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
        #Default parameter file names
        defaultpath = os.path.dirname(__file__)
        self.gadgetdefaultparam = os.path.join(defaultpath,"gadgetparams.param")
        self.gadgetparam = "gadget3.param"
        #Executable names
        self.gadgetexe = "P-Gadget3"
        self.gadgetconfig = "Config.sh"
        self.gadget_dir = os.path.expanduser("~/codes/P-Gadget3/")
        #For repeatability, we store git hashes of Gadget, GenIC, CAMB and ourselves
        #at time of running.
        self.simulation_git = utils.get_git_hash(os.path.dirname(__file__))

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
        config.write(prefix+"SFR\n")
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
            jsonstr = jsonpickle.encode(self.__dict__)
            jsout.write(jsonstr)

    def load_txt_description(self):
        """Load the text file describing the parameters of the code that generated a simulation."""
        with open(os.path.join(self.outdir, "Simulation.json"), 'r') as jsin:
            jsonstr = jsin.read()
            self.__dict__ = jsonpickle.decode(jsonstr)

    def make_simulation(self, genic_output):
        """Wrapper function to make all the simulation parameter files in turn."""
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
        self.gadget_git = utils.get_git_hash(gadget_binary)
        self.make_output = subprocess.check_output(["make", "-j8"], cwd=self.gadget_dir, universal_newlines=True)
        #Check that the last-changed time of the binary has actually changed..
        assert g_mtime != os.stat(gadget_binary).st_mtime
