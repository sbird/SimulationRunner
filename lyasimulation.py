"""Specialization of the Simulation class to Lyman-alpha forest simulations."""

import simulation

class LymanAlphaSim(simulation.Simulation):
    """Specialise the Simulation class for the Lyman alpha forest.
        This uses the QuickLya star formation module."""
    def __init__(self, outdir, box, npart, nproc, memory, timelimit, rescale_gamma = False, rescale_amp = 1., rescale_slope = -0.7, seed = 9281110, redshift=99, redend = 2, omegac=0.2408, omegab=0.0472, hubble=0.7, scalar_amp=2.427e-9, ns=0.97, uvb="hm"):
        #Parameters of the heating rate rescaling to account for helium reionisation
        #Default parameters do nothing
        self.rescale_gamma = rescale_gamma
        self.rescale_amp = rescale_amp
        self.rescale_slope = rescale_slope
        simulation.Simulation.__init__(self, outdir=outdir, box=box, npart=npart, nproc=nproc, memory = memory, timelimit=timelimit, seed=seed, redshift=redshift, redend=redend, separate_gas=True, omegac=omegac, omegab=omegab, hubble=hubble, scalar_amp=scalar_amp, ns=ns, uvb=uvb)

    def _feedback_config_options(self, config):
        """Config options specific to the Lyman alpha forest star formation criterion"""
        config.write("USE_SFR")
        config.write("QUICK_LYALPHA")
        if self.rescale_gamma:
            config.write("RESCALE_EEOS")
        return

    def _feedback_params(self, config):
        """Config options specific to the lyman alpha forest"""
        #These are parameters for the model to rescale the temperature-density relation
        if self.rescale_gamma:
            config["ExtraHeatingThresh"] = 10.0
            config["ExtraHeatingAmp"]  = self.rescale_amp
            config["ExtraHeatingExponent"] = self.rescale_slope
        return config
