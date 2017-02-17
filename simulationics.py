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
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
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
    separate_nu - if true the power spectrum check will run for neutrino particles as well (actual ICs are not changed).
    omegab - baryon density. Note that if we do not have gas particles, still set omegab, but set separate_gas = False
    omegam - Matter density
    hubble - Hubble parameter, h, which is H0 / (100 km/s/Mpc)
    scalar_amp - A_s at k = 2e-3, comparable to the WMAP value.
    ns - Scalar spectral index
    """
    def __init__(self, *, outdir, box, npart, seed = 9281110, redshift=99, separate_gas=True, separate_nu=False, omegac=0.2408, omegab=0.0472, omeganu=0.,hubble=0.7, scalar_amp=2.427e-9, ns=0.97, code_class=simulation.Simulation, code_args=None):
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
        assert omegac <= 1 and omegac > 0
        self.omegac = omegac
        assert omegab > 0 and omegab < 1
        self.omegab = omegab
        assert omeganu >=0 and omeganu < omegac
        self.omeganu = omeganu
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
        #Neutrinos: note this only affects the power spectrum check.
        #Altering this does not change ICs
        self.separate_nu = separate_nu
        self.outdir = outdir
        defaultpath = os.path.dirname(__file__)
        #Default values for the CAMB parameters
        self.cambdefault = os.path.join(defaultpath,"params.ini")
        #Filename for new CAMB file
        self.cambout = "_camb_params.ini"
        #Default GenIC paths
        self.genicdefault = os.path.join(defaultpath,"ngenic.param")
        self.genicout = "_genic_params.ini"
        #Executable names
        self.cambexe = "camb"
        self.genicexe = "N-GenIC"
        #Number of files per snapshot
        #This is chosen to give a reasonable number and
        #a constant number of particles per file.
        self.numfiles = int(np.max([2,self.npart**3//2**24]))
        #Total matter density
        self.omega0 = self.omegac + self.omegab + self.omeganu
        #Extra redshifts at which to generate CAMB output, in addition to self.redshift and self.redshift/2
        self.camb_times = [9,4,2,1,0]
        #Class with which to generate ICs.
        self.code_class_name = code_class
        #Format in which to output initial conditions: derived from Simulation class.
        self.icformat = code_class.icformat
        assert 4 >= self.icformat >= 2

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
        #Total matter density, not CDM matter density.
        config['Omega'] = self.omega0
        config['OmegaLambda'] = 1- self.omega0
        config['OmegaBaryon'] = self.omegab
        config['OmegaNeutrino'] = self.omeganu
        config['HubbleParam'] = self.hubble
        config['Redshift'] = self.redshift
        config['FileWithInputSpectrum'] = camb_output + "_matterpow_"+str(self.redshift)+".dat"
        config['FileWithTransfer'] = camb_output + "_transfer_"+str(self.redshift)+".dat"
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
        camb_file = camb_output+"_matterpow_"+str(self.redshift)+".dat"
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
        genpk = utils.find_exec("gen-pk")
        genicfileout = os.path.join(self.outdir, genicfileout)
        subprocess.check_call([genpk, "-i", genicfileout, "-o", os.path.dirname(genicfileout)])
        #Now check that they match what we put into the simulation, from CAMB
        #Reload the CAMB files from disc, just in case something went wrong writing them.
        matterpow = camb_output + "_matterpow_"+str(self.redshift)+".dat"
        transfer = camb_output + "_transfer_"+str(self.redshift)+".dat"
        camb = cambpower.CAMBPowerSpectrum(matterpow, transfer, kmin=2*math.pi/self.box/5, kmax = self.npart*2*math.pi/self.box*10)
        #Error to tolerate on simulated power spectrum
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
                accuracy *= 2
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

    def make_simulation(self, pkaccuracy=0.05):
        """Wrapper function to make the simulation ICs."""
        #First generate the input files for CAMB
        (camb_output, camb_param) = self.cambfile()
        #Then run CAMB
        camb = utils.find_exec(self.cambexe)
        self.camb_git = utils.get_git_hash(camb)
        #In python 3.5, can use subprocess.run to do this.
        #But for backwards compat, use check_output
        subprocess.check_call([os.path.join(os.getcwd(), camb), camb_param], cwd=os.path.dirname(camb))
        #Change the power spectrum file on disc if we want to do that
        self._alter_power(camb_output)
        #Now generate the GenIC parameters
        (genic_output, genic_param) = self.genicfile(camb_output)
        #Run N-GenIC
        genic = utils.find_exec(self.genicexe)
        self.genic_git = utils.get_git_hash(genic)
        subprocess.check_call([genic, genic_param])
        #Save a json of ourselves.
        self.txt_description()
        #Check that the ICs have the right power spectrum
        self.check_ic_power_spectra(camb_output, genic_output,accuracy=pkaccuracy)
        #Make the parameter files.
        ics = self.code_class_name(outdir=self.outdir, box=self.box, npart=self.npart, redshift=self.redshift, separate_gas=self.separate_gas, omegac=self.omegac, omegab=self.omegab, omeganu=self.omeganu, hubble=self.hubble, **self.code_args)
        return ics.make_simulation(genic_output)

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
