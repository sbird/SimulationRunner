"""Specialization of the Simulation class to Lyman-alpha forest simulations."""

import numpy as np
from . import mpsimulation
from . import simulationics

class NeutrinoPartSim(mpsimulation.MPSimulation):
    """Further specialise the Simulation class for particle based massive neutrinos.
    """
    __doc__ = __doc__+mpsimulation.MPSimulation.__doc__
    def _other_params(self, config):
        """Config options to make type 2 particles neutrinos."""
        config['FastParticleType'] = 2
        #Neutrinos move quickly, so we must rebuild
        #the tree every time step.
        config['TreeDomainUpdateFrequency'] = 0.0
        #Specify neutrino masses so that omega_nu is correct
        numass = _get_neutrino_masses(self.m_nu, 0)
        config['MNue'] = numass[2]
        config['MNum'] = numass[1]
        config['MNut'] = numass[0]
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

    def _camb_neutrinos(self, config):
        """Config options so CAMB can use massive neutrinos.
        For particle neutrinos we want degenerate neutrinos."""
        config['massless_neutrinos'] = 0.046
        config['massive_neutrinos'] = 3
        config['nu_mass_fractions'] = 1
        #Very light neutrinos should have only one massive species
        if self.m_nu < 0.1:
            config['massless_neutrinos'] = 2.046
            config['massive_neutrinos'] = 1
        config['nu_mass_eigenstates'] = 1
        #Actually does nothing, but we set it to avoid the library setting it to ""
        config['nu_mass_degeneracies'] = 0
        config['share_delta_neff'] = 'T'
        #Set the neutrino density and subtract it from omega0
        omeganuh2 = self.m_nu/93.14
        config['omnuh2'] = omeganuh2
        config['omch2'] = (self.omega0 - self.omegab)*self.hubble**2- omeganuh2
        return config

    def _genicfile_child_options(self, config):
        """Set up particle neutrino parameters for GenIC"""
        config['NU_Vtherm_On'] = 1
        config['NNeutrino'] = self.npart
        config['NU_PartMass_in_ev'] = self.m_nu
        #Degenerate neutrinos
        config['Hierarchy'] = 0
        return config

def _get_neutrino_masses(total_mass, hierarchy):
    """Get the three neutrino masses, including the mass splittings.
        Hierarchy is -1 for inverted (two heavy), 1 for normal (two light) and 0 for degenerate."""
    #Neutrino mass splittings
    nu_M21 = 7.53e-5 #Particle data group 2016: +- 0.18e-5 eV2
    nu_M32n = 2.44e-3 #Particle data group: +- 0.06e-3 eV2
    nu_M32i = 2.51e-3 #Particle data group: +- 0.06e-3 eV2

    if hierarchy > 0:
        nu_M32 = nu_M32n
        #If the total mass is below that allowed by the hierarchy,
        #assign one active neutrino.
        if total_mass < np.sqrt(nu_M32n) + np.sqrt(nu_M21):
            return np.array([total_mass, 0, 0])
    elif hierarchy < 0:
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

class NeutrinoSemiLinearSim(mpsimulation.MPSimulation):
    """Further specialise the Simulation class for semi-linear analytic massive neutrinos.
    """
    def __init__(self, *, hierarchy=0, **kwargs):
        self.hierarchy = hierarchy
        super().__init__(**kwargs)

    def _other_params(self, config):
        """Config options specific to kspace neutrinos, which computes neutrino hierarchy."""
        config['MassiveNuLinRespOn'] = 1
        numass = _get_neutrino_masses(self.m_nu, self.hierarchy)
        config['MNue'] = numass[2]
        config['MNum'] = numass[1]
        config['MNut'] = numass[0]
        config['LinearTransferFunction'] = "camb_linear/ics_transfer_"+str(self.redshift)+".dat"
        config['InputSpectrum_UnitLength_in_cm'] = 3.085678e24
        config['TimeTransfer'] = 1./(1+self.redshift)
        return config

class NeutrinoSemiLinearICs(NeutrinoPartICs):
    """Further specialise the NeutrinoPartICs class for semi-linear analytic massive neutrinos.
    """
    __doc__ = __doc__+NeutrinoPartICs.__doc__
    def __init__(self, *, hierarchy = 0, code_class=NeutrinoSemiLinearSim, code_args = None, **kwargs):
        if code_args is not None:
            code_args['hierarchy'] = hierarchy
        else:
            code_args = {'hierarchy': hierarchy}
        self.hierarchy = hierarchy
        super().__init__(code_class=code_class, code_args=code_args, **kwargs)

    def _genicfile_child_options(self, config):
        """Set up neutrino parameters for GenIC.
        This just includes a change in OmegaNu, but no actual particles."""
        config['NNeutrino'] = 0
        config['NU_in_DM'] = 0
        config['NU_PartMass_in_ev'] = self.m_nu
        #Degenerate neutrinos
        config['Hierarchy'] = self.hierarchy
        return config

    def _camb_neutrinos(self, config):
        """Config options so CAMB can use massive neutrinos.
        We want a neutrino mass hierarchy."""
        config['massless_neutrinos'] = 0.046
        config['massive_neutrinos'] = [1, 1, 1]
        numass = _get_neutrino_masses(self.m_nu, self.hierarchy)
        config['nu_mass_fractions'] = list(numass/self.m_nu)
        config['nu_mass_eigenstates'] = 3
        #Each neutrino has the same temperature
        config['share_delta_neff'] = 'T'
        #Actually does nothing if share_delta_neff = T,q
        #but we set it to avoid the library setting it to "" which CAMB's parser rejects.
        config['nu_mass_degeneracies'] = 0
        #Set the neutrino density and subtract it from omega0
        omeganuh2 = self.m_nu/93.14
        config['omnuh2'] = omeganuh2
        config['omch2'] = (self.omega0 - self.omegab)*self.hubble**2- omeganuh2
        return config

class NeutrinoHybridSim(NeutrinoSemiLinearSim):
    """Further specialise to hybrid neutrino simulations, which have both analytic and particle neutrinos."""
    __doc__ = __doc__+mpsimulation.MPSimulation.__doc__
    def __init__(self, *, zz_transition=1., vcrit=500, **kwargs):
        super().__init__(**kwargs)
        self.vcrit = vcrit
        self.zz_transition = zz_transition

    def _other_params(self, config):
        """Config options specific to kspace neutrinos. Hierarchy is off for now."""
        config = NeutrinoSemiLinearSim._other_params(self, config)
        config['HybridNeutrinosOn'] = 1
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
        config['NNeutrino'] = int(self.npart*self.npartnufac)
        config['NU_PartMass_in_ev'] = self.m_nu
        config['NU_in_DM'] = 0
        config['Max_nuvel'] = self.vcrit
        return config
