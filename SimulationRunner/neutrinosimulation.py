"""Specialization of the Simulation class to Lyman-alpha forest simulations."""

import numpy as np
from . import simulation
from . import simulationics

class NeutrinoPartSim(simulation.Simulation):
    """Further specialise the Simulation class for particle based massive neutrinos.
    """
    __doc__ = __doc__+simulation.Simulation.__doc__
    def _other_params(self, config):
        """Config options to make type 2 particles neutrinos."""
        #Specify neutrino masses so that omega_nu is correct
        config['MassiveNuLinRespOn'] = 0
        return config

class NeutrinoPartICs(simulationics.SimulationICs):
    """Specialise the initial conditions for particle neutrinos."""
    __doc__ = __doc__+simulationics.SimulationICs.__doc__
    def __init__(self, *, m_nu=0.1, separate_gas=False, code_class=NeutrinoPartSim, **kwargs):
        #Set neutrino mass
        #Note that omega0 does remains constant if we change m_nu.
        #This does mean that omegab/omegac will increase, but not by much.
        assert m_nu > 0
        super().__init__(m_nu = m_nu, separate_gas=separate_gas, code_class=code_class, **kwargs)
        self.separate_nu = True

    def _genicfile_child_options(self, config):
        """Set up particle neutrino parameters for GenIC"""
        config['NgridNu'] = self.npart
        config['MNue'] = self.m_nu/3.
        config['MNum'] = self.m_nu/3.
        config['MNut'] = self.m_nu/3.
        #Degenerate neutrinos
        return config

def get_neutrino_masses(total_mass, hierarchy):
    """Get the three neutrino masses, including the mass splittings.
        Hierarchy is 'inverted' (two heavy), 'normal' (two light) or degenerate."""
    #Neutrino mass splittings
    nu_M21 = 7.53e-5 #Particle data group 2016: +- 0.18e-5 eV2
    nu_M32n = 2.44e-3 #Particle data group: +- 0.06e-3 eV2
    nu_M32i = 2.51e-3 #Particle data group: +- 0.06e-3 eV2

    if hierarchy == 'normal':
        nu_M32 = nu_M32n
        #If the total mass is below that allowed by the hierarchy,
        #assign one active neutrino.
        if total_mass < np.sqrt(nu_M32n) + np.sqrt(nu_M21):
            return np.array([total_mass, 0, 0])
    elif hierarchy == 'inverted':
        nu_M32 = -nu_M32i
        if total_mass < 2*np.sqrt(nu_M32i) - np.sqrt(nu_M21):
            return np.array([total_mass/2., total_mass/2., 0])
    #Hierarchy == 0 is 3 degenerate neutrinos
    else:
        return np.ones(3)*total_mass/3.

    #DD is the summed masses of the two closest neutrinos
    DD1 = 4 * total_mass/3. - 2/3.*np.sqrt(total_mass**2 + 3*nu_M32 + 1.5*nu_M21)
    #Last term was neglected initially. This should be very well converged.
    DD = 4 * total_mass/3. - 2/3.*np.sqrt(total_mass**2 + 3*nu_M32 + 1.5*nu_M21+0.75*nu_M21**2/DD1**2)
    nu_masses = np.array([ total_mass - DD, 0.5*(DD + nu_M21/DD), 0.5*(DD - nu_M21/DD)])
    assert np.isfinite(DD)
    assert np.abs(DD1/DD -1) < 2e-2
    assert np.all(nu_masses >= 0)
    return nu_masses

class NeutrinoSemiLinearSim(simulation.Simulation):
    """Further specialise the Simulation class for semi-linear analytic massive neutrinos.
    """
    def __init__(self, *, nu_hierarchy='normal', **kwargs):
        super().__init__(**kwargs)
        self.nu_hierarchy = nu_hierarchy

    def _other_params(self, config):
        """Config options specific to kspace neutrinos, which computes neutrino hierarchy."""
        config['MassiveNuLinRespOn'] = 1
        numass = get_neutrino_masses(self.m_nu, self.nu_hierarchy)
        config['MNue'] = numass[2]
        config['MNum'] = numass[1]
        config['MNut'] = numass[0]
        config['LinearTransferFunction'] = "camb_linear/ics_transfer_"+str(self.redshift)+".dat"
        config['InputSpectrum_UnitLength_in_cm'] = 3.085678e24
        return config

class NeutrinoSemiLinearICs(NeutrinoPartICs):
    """Further specialise the NeutrinoPartICs class for semi-linear analytic massive neutrinos.
    """
    __doc__ = __doc__+NeutrinoPartICs.__doc__
    def __init__(self, *, code_class=NeutrinoSemiLinearSim, **kwargs):
        super().__init__(code_class=code_class, **kwargs)

    def _genicfile_child_options(self, config):
        """Set up neutrino parameters for GenIC.
        This just includes a change in OmegaNu, but no actual particles."""
        config['NgridNu'] = 0
        numass = get_neutrino_masses(self.m_nu, self.nu_hierarchy)
        config['MNue'] = numass[2]
        config['MNum'] = numass[1]
        config['MNut'] = numass[0]
        #Degenerate neutrinos
        return config

class NeutrinoHybridSim(NeutrinoSemiLinearSim):
    """Further specialise to hybrid neutrino simulations, which have both analytic and particle neutrinos."""
    __doc__ = __doc__+simulation.Simulation.__doc__
    def __init__(self, *, zz_transition=1., vcrit=500, nu_hierarchy = 'degenerate', **kwargs):
        super().__init__(nu_hierarchy=nu_hierarchy, **kwargs)
        self.vcrit = vcrit
        self.zz_transition = zz_transition

    def _other_params(self, config):
        """Config options specific to kspace neutrinos. Hierarchy is off for now."""
        config = NeutrinoSemiLinearSim._other_params(self, config)
        config['HybridNeutrinosOn'] = 1
        config['FastParticleType'] = 2
        config['TreeDomainUpdateFrequency'] = 0.0
        config['Vcrit'] = self.vcrit
        config['NuPartTime'] = 1./(1+self.zz_transition)
        return config

class NeutrinoHybridICs(NeutrinoPartICs):
    """Further specialise the NeutrinoPartICs class for semi-linear analytic massive neutrinos.
    """
    def __init__(self, *, npartnufac=0.5, vcrit=500, code_class=NeutrinoHybridSim, code_args = None, **kwargs):
        self.npartnufac = npartnufac
        self.vcrit = vcrit
        ncode_args = {'vcrit':vcrit, 'zz_transition': 1.}
        if code_args is not None:
            ncode_args.update(code_args)
        super().__init__(code_class=code_class, code_args=ncode_args, **kwargs)

    def _genicfile_child_options(self, config):
        """Set up hybrid neutrino parameters for GenIC."""
        config['NU_Vtherm_On'] = 1
        numass = get_neutrino_masses(self.m_nu, self.nu_hierarchy)
        config['MNue'] = numass[2]
        config['MNum'] = numass[1]
        config['MNut'] = numass[0]
        #Degenerate neutrinos
        config['NgridNu'] = int(self.npart*self.npartnufac)
        config['Max_nuvel'] = self.vcrit
        return config
