"""Specialization of the Simulation class to Lyman-alpha forest simulations."""

import simulation
import os.path
import numpy as np

class LymanAlphaSim(simulation.Simulation):
    """Specialise the Simulation class for the Lyman alpha forest.
        This uses the QuickLya star formation module."""
    def __init__(self, outdir, box, npart, nproc, memory, timelimit, rescale_gamma = False, rescale_amp = 1., rescale_slope = -0.7, seed = 9281110, redshift=99, redend = 2, omegac=0.2408, omegab=0.0472, hubble=0.7, scalar_amp=2.427e-9, ns=0.97, uvb="hm"):
        #Parameters of the heating rate rescaling to account for helium reionisation
        #Default parameters do nothing
        self.rescale_gamma = rescale_gamma
        self.rescale_amp = rescale_amp
        self.rescale_slope = rescale_slope
        #TODO: Set up new output directory hierarchy for Lyman alpha simulations
        new_outdir = outdir
        simulation.Simulation.__init__(self, outdir=new_outdir, box=box, npart=npart, nproc=nproc, memory = memory, timelimit=timelimit, seed=seed, redshift=redshift, redend=redend, separate_gas=True, omegac=omegac, omegab=omegab, hubble=hubble, scalar_amp=scalar_amp, ns=ns, uvb=uvb)
        self.camb_times = [9,]+[x for x in np.arange(4.2,1.9,-0.2)]


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

    def _generate_times(self):
        """Snapshot outputs for lyman alpha"""
        redshifts = np.concatenate([[49,9],np.arange(4.2,1.9,-0.2)])
        return 1./(1.+redshifts)

#TODO: memory, timelimit and nproc should be moved to machine specific properties in a decorator.
if __name__ == "__main__":
    ss = LymanAlphaSim(os.path.expanduser("~/data/Lya_Boss/test1"), box=60, npart=512, nproc=256,memory=2000,timelimit=24.)
    ss.make_simulation(verbose=True)
