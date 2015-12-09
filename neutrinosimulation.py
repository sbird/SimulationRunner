"""Specialization of the Simulation class to Lyman-alpha forest simulations."""

import simulation

class NeutrinoSim(simulation.Simulation):
    """Specialise the Simulation class for massive neutrinos.
    """
    __doc__ = __doc__+simulation.Simulation.__doc__
    def __init__(self, outdir, box, npart, *, m_nu = 0., hubble=0.7, omegac=0.2408, **kwargs):
        #Set neutrino mass
        omeganu = 3*m_nu/93.14/hubble/hubble
        self.m_nu = m_nu
        #Subtract omeganu from omegac, so that with changing
        #neutrino mass the total matter fraction remains constant.
        #Note this does mean that omegab/omegac will increase, but not by much.
        omegac = omegac-omeganu
        simulation.Simulation.__init__(self, outdir=outdir, box=box, npart=npart, omegac=omegac, omeganu=omeganu, separate_nu=True, hubble=hubble, **kwargs)

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

    def _gadget3_child_options(self, config):
        """Config options to turn on the right neutrino method"""
        config.write("NEUTRINOS\n")
        return

class NeutrinoPartSim(NeutrinoSim):
    """Further specialise the Simulation class for particle based massive neutrinos.
    """
    __doc__ = __doc__+simulation.Simulation.__doc__
    def _genicfile_child_options(self, config):
        """Set up particle neutrino parameters for GenIC"""
        config['NU_On'] = 1
        config['NU_Vtherm_On'] = 1
        config['NNeutrino'] = self.npart
        config['NU_PartMass_in_ev'] = self.m_nu
        return config

    def _gadget3_child_options(self, config):
        """Config options to turn on the right neutrino method"""
        config.write("NEUTRINOS\n")
        return

class NeutrinoSemiLinearSim(NeutrinoSim):
    """Further specialise the Simulation class for semi-linear analytic massive neutrinos.
    """
    __doc__ = __doc__+simulation.Simulation.__doc__
    def _genicfile_child_options(self, config):
        """Set up neutrino parameters for GenIC.
        This just includes a change in OmegaNu, but no actual particles."""
        config['NNeutrino'] = 0
        config['NU_KSPACE'] = 0
        return config

    def _gadget3_child_options(self, config):
        """Config options to turn on the right neutrino method"""
        #Note for some Gadget versions we may need KSPACE_NEUTRINOS
        config.write("KSPACE_NEUTRINOS_2\n")
        return

    def _other_params(self, config):
        """Config options specific to kspace neutrinos. Hierarchy is off for now."""
        config['MNue'] = self.m_nu/3.
        config['MNum'] = self.m_nu/3.
        config['MNut'] = self.m_nu/3.
        config['KspaceTransferFunction'] = "ics_transfer_"+str(self.redshift)+".dat"
        config['InputSpectrum_UnitLength_in_cm'] = 3.085678e24
        config['TimeTransfer'] = 1./(1+self.redshift)
        config['OmegaBaryonCAMB'] = self.omegab
        return config

class NeutrinoHybridSim(NeutrinoSim):
    """Further specialise to hybrid neutrino simulations, which have both analytic and particle neutrinos."""
    __doc__ = __doc__+simulation.Simulation.__doc__
    def __init__(self, outdir, box, npart, *, m_nu = 0., vcrit=500, npartnufac = 0.5, zz_transition=1., **kwargs):
        NeutrinoSim.__init__(self, outdir=outdir, box=box, npart=npart, m_nu=m_nu, **kwargs)
        self.vcrit = vcrit
        self.zz_transition = zz_transition
        self.npartnufac = npartnufac

    def _genicfile_child_options(self, config):
        """Set up neutrino parameters for GenIC.
        This just includes a change in OmegaNu, but no actual particles."""
        config['NU_On'] = 1
        config['NU_Vtherm_On'] = 1
        config['NNeutrino'] = int(self.npart*self.npartnufac)
        config['Nu_PartMass_in_ev'] = self.m_nu
        config['NU_KSPACE'] = 0
        config['Max_nuvel'] = self.vcrit
        return config

    def _gadget3_child_options(self, config):
        """Config options to turn on the right neutrino method"""
        #Note for some Gadget versions we may need KSPACE_NEUTRINOS
        config.write("KSPACE_NEUTRINOS_2\n")
        config.write("HYBRID_NEUTRINOS\n")
        return

    def _other_params(self, config):
        """Config options specific to kspace neutrinos. Hierarchy is off for now."""
        config['MNue'] = self.m_nu/3.
        config['MNum'] = self.m_nu/3.
        config['MNut'] = self.m_nu/3.
        config['KspaceTransferFunction'] = "ics_transfer_"+str(self.redshift)+".dat"
        config['InputSpectrum_UnitLength_in_cm'] = 3.085678e24
        config['TimeTransfer'] = 1./(1+self.redshift)
        config['OmegaBaryonCAMB'] = self.omegab
        config['VCRIT'] = self.vcrit
        config['NuPartTime'] = 1./(1+self.zz_transition)
        return config

