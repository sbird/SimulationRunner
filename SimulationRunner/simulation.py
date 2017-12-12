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
import shutil
import subprocess
import configobj
import numpy as np
from . import read_uvb_tab
from . import utils
from . import clusters

class Simulation(object):
    """
    Class for creating config files for MP-Gadget to run a single simulation.
    ICs are generated in simulationics.py

    Init parameters:
    outdir - Directory in which to save ICs
    box - Box size in comoving Mpc/h
    npart - Cube root of number of particles
    separate_gas - if true the ICs will contain baryonic particles. If false, just DM.
    redshift - redshift at which to generate ICs
    redend - redshift to run simulation to
    m_nu - neutrino mass, used by child classes.
    omegab - baryon density. Note that if we do not have gas particles, still set omegab, but set separate_gas = False
    omega0 - Total matter density
    hubble - Hubble parameter, h, which is H0 / (100 km/s/Mpc)
    uvb - Ultra-violet background to use
    """
    icformat = 4
    def __init__(self, *, outdir, box, npart, redshift=99, redend = 0, m_nu = 0, separate_gas=True, omega0=0.288, omegab=0.0472, hubble=0.7, uvb="hm", cluster_class=clusters.MARCCClass):
        #Check that input is reasonable and set parameters
        #In Mpc/h
        assert box < 20000
        self.box = box
        #Cube root
        assert npart > 1 and npart < 16000
        self.npart = int(npart)
        #Physically reasonable
        assert omegab > 0 and omegab < 1
        self.omegab = omegab
        #Total matter density
        self.omega0 = omega0
        assert 0 < self.omega0 <= 1
        assert redshift > 1 and redshift < 1100
        self.redshift = redshift
        assert redend >= 0 and redend < 1100
        self.redend = redend
        assert hubble < 1 and hubble > 0
        self.hubble = hubble
        #Neutrino mass: used by child classes
        self.m_nu = m_nu
        #Baryons?
        self.separate_gas = separate_gas
        #UVB? Only matters if gas
        self.uvb = uvb
        assert self.uvb == "hm" or self.uvb == "fg" or self.uvb == "sh"
        #Number of files per snapshot
        #This is chosen to give a reasonable number and
        #a constant number of particles per file.
        self.numfiles = int(np.max([2,self.npart**3//2**24]))
        outdir = os.path.realpath(os.path.expanduser(outdir))
        assert os.path.exists(outdir)
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
        #Executable names
        self.gadgetexe = "MP-Gadget"
        self.gadgetconfig = "Options.mk"
        self.gadget_dir = os.path.expanduser("~/codes/MP-Gadget/")
        self.gadget_binary_dir = os.path.join(self.gadget_dir,"build")

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

    def generate_mpi_submit(self):
        """Generate a sample mpi_submit file.
        The prefix argument is a string at the start of each line.
        It separates queueing system directives from normal comments"""
        self._cluster.generate_mpi_submit(self.outdir)

    def make_simulation(self, genic_output, do_build=False):
        """Wrapper function to make all the simulation parameter files in turn."""
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
            self.make_output = subprocess.check_output(["make", "-j8"], cwd=self.gadget_dir, universal_newlines=True, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise
        #Check that the last-changed time of the binary has actually changed..
        assert g_mtime != os.stat(gadget_binary).st_mtime
        shutil.copy(gadget_binary, os.path.join(os.path.dirname(gadget_config),self.gadgetexe))
