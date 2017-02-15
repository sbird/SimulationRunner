"""Specialization of the Simulation class to Lyman-alpha forest simulations."""

from . import mpsimulation
from . import simulationics

class NeutrinoPartSim(mpsimulation.MPSimulation):
    """Further specialise the Simulation class for particle based massive neutrinos.
    """
    __doc__ = __doc__+mpsimulation.MPSimulation.__doc__
    def _other_params(self, config):
        """Config options to make type 2 particles neutrinos."""
        config['FastParticleType'] = 2
        config['NoTreeType'] = 2
        return config

class NeutrinoPartICs(simulationics.SimulationICs):
    """Specialise the initial conditions for particle neutrinos."""
    __doc__ = __doc__+simulationics.SimulationICs.__doc__
    def __init__(self, *, m_nu=0.1, hubble=0.7, omegac=0.2408, separate_gas=False, code_class=NeutrinoPartSim, **kwargs):
        #Set neutrino mass
        assert m_nu > 0
        omeganu = m_nu/93.14/hubble/hubble
        self.m_nu = m_nu
        #Subtract omeganu from omegac, so that with changing
        #neutrino mass the total matter fraction remains constant.
        #Note this does mean that omegab/omegac will increase, but not by much.
        omegac = omegac-omeganu
        super().__init__(omegac=omegac, omeganu=omeganu, hubble=hubble, separate_gas=separate_gas, separate_nu=True, code_class=code_class, **kwargs)

    def _camb_neutrinos(self, config):
        """Config options so CAMB can use massive neutrinos.
        For particle neutrinos we want to neglect hierarchy."""
        config['massless_neutrinos'] = 0.046
        config['massive_neutrinos'] = 3
        config['nu_mass_fractions'] = 1
        config['nu_mass_eigenstates'] = 1
        #Actually does nothing, but we set it to avoid the library setting it to ""
        config['nu_mass_degeneracies'] = 0
        config['share_delta_neff'] = 'T'
        return config

    def _genicfile_child_options(self, config):
        """Set up particle neutrino parameters for GenIC"""
        config['NU_Vtherm_On'] = 1
        config['NNeutrino'] = self.npart
        config['NU_PartMass_in_ev'] = self.m_nu
        return config

class NeutrinoSemiLinearSim(mpsimulation.MPSimulation):
    """Further specialise the Simulation class for semi-linear analytic massive neutrinos.
    """
    def __init__(self, *, m_nu=0.1, **kwargs):
        self.m_nu = m_nu
        super().__init__(**kwargs)

    def _other_params(self, config):
        """Config options specific to kspace neutrinos. Hierarchy is off for now."""
        config['MassiveNuLinRespOn'] = 1
        config['MNue'] = self.m_nu/3.
        config['MNum'] = self.m_nu/3.
        config['MNut'] = self.m_nu/3.
        config['KspaceTransferFunction'] = "camb_linear/ics_transfer_"+str(self.redshift)+".dat"
        config['InputSpectrum_UnitLength_in_cm'] = 3.085678e24
        config['TimeTransfer'] = 1./(1+self.redshift)
        config['OmegaBaryonCAMB'] = self.omegab
        return config

class NeutrinoSemiLinearICs(NeutrinoPartICs):
    """Further specialise the NeutrinoPartICs class for semi-linear analytic massive neutrinos.
    """
    __doc__ = __doc__+NeutrinoPartICs.__doc__
    def __init__(self, *, m_nu = 0.1, code_class=NeutrinoSemiLinearSim, code_args = None, **kwargs):
        if code_args is not None:
            code_args['m_nu'] = m_nu
        else:
            code_args = {'m_nu':m_nu}
        super().__init__(m_nu = m_nu, code_class=code_class, code_args=code_args, **kwargs)

    def _genicfile_child_options(self, config):
        """Set up neutrino parameters for GenIC.
        This just includes a change in OmegaNu, but no actual particles."""
        config['NNeutrino'] = 0
        config['NU_in_DM'] = 0
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
