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
import configobj
from . import simulation

class MPSimulation(simulation.Simulation):
    """
    Class for creating config files needed to run MP-Gadget3 in a single simulation.

    Init parameters:
    outdir - Directory in which to save ICs
    box - Box size in comoving Mpc/h
    npart - Cube root of number of particles
    separate_gas - if true the ICs will contain baryonic particles. If false, just DM.
    redshift - redshift at which to generate ICs
    omegab - baryon density. Note that if we do not have gas particles, still set omegab, but set separate_gas = False
    omegam - Matter density
    hubble - Hubble parameter, h, which is H0 / (100 km/s/Mpc)
    """
    icformat = 4
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
            config.write("GSL_INCL = $(shell pkg-config --cflags gsl)\nGSL_LIBS = $(shell pkg-config --libs gsl)\n")
            #We may want DENSITY_INDEPENDENT_SPH as well.
            #config.write(prefix+"DENSITY_INDEPENDENT_SPH\n")
            config.write(prefix+"OPENMP_USE_SPINLOCK\n")
            config.write(prefix+"TOPNODEFACTOR=5.0\n")
            config.write(prefix+"INHOMOG_GASDISTR_HINT\n")
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
        config.write(prefix+"WINDS\n")
        config.write(prefix+"METALS\n")
        config.write(prefix+"BLACKHOLES\n")
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
        config['SnapshotFileBase'] = "snap"
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
        #This should just be larger than the simulation time limit
        config['CpuTimeBetRestartFile'] = 60*60*self._cluster.timelimit*10
        #Softening is 1/30 of the mean linear interparticle spacing
        soften = 1000 * self.box/self.npart/30.
        for ptype in ('Gas', 'Halo', 'Disk', 'Bulge', 'Stars', 'Bndry'):
            config['Softening'+ptype] = soften
            config['Softening'+ptype+'MaxPhys'] = soften
        #This could be tuned in lower memory conditions
        config['BufferSize'] = 100
        #These are only used for gas, but must be set anyway
        config['MinGasTemp'] = 100
        #In equilibrium with the CMB at early times.
        config['InitGasTemp'] = 2.7*(1+self.redshift)
        #Set the required neutrino parameters.
        config['MassiveNuLinRespOn'] = 0
        config['LinearTransferFunction'] = "camb_linear/ics_transfer_"+str(self.redshift)+".dat"
        config['TimeTransfer'] = 1./(1+self.redshift)
        #This needs to be here until I fix the flux extractor to allow quintic kernels.
        config['DensityKernelType'] = 'cubic'
        if self.separate_gas:
            config['CoolingOn'] = 1
            config['TreeCoolFile'] = "TREECOOL"
            #Copy a TREECOOL file into the right place.
            self._copy_uvb()
            #Need more memory for a feedback model
            config['PartAllocFactor'] = 4
            config = self._sfr_params(config)
            config = self._feedback_params(config)
        else:
            config['CoolingOn'] = 0
            config['StarformationOn'] = 0
            config['PartAllocFactor'] = 2
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
        config['WindModel'] = 'nowind'
        return config

    def _other_params(self, config):
        """Function to override to set other config parameters"""
        return config
