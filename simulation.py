"""
Module to automate the generation of simulation config files.
The base class is Simulation, which creates the config files for a single simulation.
It is meant to be called from other classes as part of a suite,
More specialised simulation types can inherit from it.
Different machines can be implemented as decorators.
"""
import os.path
import re
import configobj
import math

def _scan_file_set(config_lines, paramname, paramval, separator):
    r"""Scan a list of lines for a particular pattern and replace it with another.
        Used to change an individual parameter value.
        The pattern scanned for is
        paramname+'\s*'+separator+'\s*[0-9.]
        if paramval is numeric
        or
        paramname+'\s*'+separator+'\s*\w*
        if it is a string.
    """



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
    def __init__(self, outdir, box, npart, seed = 9281110, redshift=99, separate_gas=True, omegac=0.2408, omegab=0.0472, hubble=0.7, scalar_amp=2.427e-9, ns=0.97):
        #Check that input is reasonable and set parameters
        #In Mpc/h
        assert box < 20000
        self.box = box
        #Cube root
        assert npart > 1 and npart < 16000
        self.npart = npart
        #Physically reasonable
        assert omegac <= 1 and omegac > 0
        self.omegac = omegac
        assert omegab > 0 and omegab < 1
        self.omegab = omegab
        assert redshift > 1 and redshift < 1100
        self.redshift = redshift
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
        self.omeganu = 0
        outdir = os.path.expanduser(outdir)
        #Make the output directory: will fail if parent does not exist
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        else:
            if os.listdir(outdir) != []:
                print "Warning: ",outdir," is non-empty"
        self.outdir = outdir
        #Default values for the CAMB parameters
        self.cambdefault = "params.ini"
        #Filename for new CAMB file
        self.cambout = "_camb_params.ini"
        #Default GenIC paths
        self.genicdefault = "ngenic.param"
        self.genicout = "_genic_params.ini"

    def cambfile(self):
        """Generate the CAMB parameter file from the (cosmological) simulation parameters and the default values"""
        #Load CAMB file using ConfigObj
        config = configobj.ConfigObj(self.cambdefault)
        config.filename = os.path.join(self.outdir, self.cambout)
        #Set values
        config['output_root'] = os.path.join(self.outdir,"camb_linear")+"/ics_"
        #Can't change this easily because the parameters then have different names
        assert config['use_physical'] == 'T'
        config['hubble'] = self.hubble * 100
        config['ombh2'] = self.omegab*self.hubble**2
        config['omch2'] = self.omegac*self.hubble**2
        config['omk'] = 0.
        #Initial power spectrum: MAKE SURE you set the pivot scale to the WMAP value!
        config['pivot_scalar'] = 2e-3
        config['pivot_tensor'] = 2e-3
        config['scalar_specral_index(1)'] = self.ns
        config['scalar_specral_amp(1)'] = self.scalar_amp
        #Various numerical parameters
        #Maximum relevant scale is 2 pi * softening length. Use a kmax double that for safety.
        config['transfer_kmax'] = 2*math.pi*100*self.npart/self.box
        #At which redshifts should we produce CAMB output: we want the starting redshift of the simulation,
        #but we also want some other values for checking purposes
        redshifts = [self.redshift, (self.redshift+1)/2-1] + [9,4,2,1,0]
        for (n,zz) in zip(range(len(redshifts)), redshifts):
            config['transfer_redshift('+str(n)+')'] = zz
            config['transfer_filename('+str(n)+')'] = 'transfer_'+str(zz)+'.dat'
            config['transfer_matterpower('+str(n)+')'] = 'matterpow_'+str(zz)+'.dat'
        #Set up the neutrinos.
        #This has it's own function so it can be overriden by child classes
        config = self._camb_neutrinos(config)
        #Write the config file
        config.write()

    def _camb_neutrinos(self, config):
        """Modify the CAMB config file to have massless neutrinos.
        Designed to be easily over-ridden"""
        config['massless_neutrinos'] = 3.046
        config['massive_neutrinos'] = 0
        return config

    def genicfile(self):
        """Generate the GenIC parameter file"""
        config = configobj.ConfigObj(self.genicdefault)
        config.filename = os.path.join(self.outdir, self.genicout)
        config['Box'] = self.box*1000
        config['Nsample'] = self.npart
        config['Nmesh'] = self.npart * 3/2
        config['OutputDir'] = self.outdir+"/ICS/"
        #Is this enough information, or should I add a short hash?
        config['FileBase'] = str(self.box)+"_"+str(self.npart)+"_"+str(self.redshift)
        #Whether we have baryons is entirely controlled by the glass file.
        #Since the glass file is just a regular grid, this should probably be in GenIC at some point
        if self.separate_gas:
            config['GlassFile'] = os.path.expanduser("~/data/glass/reg-grid-128-2comp")
        else:
            config['GlassFile'] = os.path.expanduser("~/data/glass/reg-grid-128-dm")
        config['GlassTileFac'] = self.npart/128
        #Total matter density, not CDM matter density.
        config['Omega'] = self.omegac + self.omegab + self.omeganu
        config['OmegaLambda'] = 1- self.omegac - self.omegab - self.omeganu
        config['OmegaBaryon'] = self.omegab
        config['OmegaDM_2ndSpecies'] = self.omeganu
        config['HubbleParam'] = self.hubble
        config['Redshift'] = self.redshift
        config['FileWithInputSpectrum'] = os.path.join(os.path.join(self.outdir, "camb_linear"), "ics_matterpow_"+str(self.redshift)+".dat")
        config['FileWithTransfer'] = os.path.join(os.path.join(self.outdir, "camb_linear"), "ics_transfer_"+str(self.redshift)+".dat")
        assert config['InputSpectrum_UnitLength_in_cm'] == '3.085678e24'
        config = self._genicfile_neutrinos(config)
        config['Seed'] = self.seed
        config.write()

    def _genicfile_neutrinos(self, config):
        """Neutrino parameters easily overridden"""
        config['NU_On'] = 0
        return config
