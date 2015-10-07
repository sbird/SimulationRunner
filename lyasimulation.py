"""Specialization of the Simulation class to Lyman-alpha forest simulations."""

import simulation
import os
import numpy as np
import string

class LymanAlphaSim(simulation.Simulation):
    """Specialise the Simulation class for the Lyman alpha forest.
        This uses the QuickLya star formation module and allows for altering the power spectrum with knots
        """
    __doc__ = __doc__+simulation.Simulation.__doc__
    def __init__(self, outdir, box, npart, knot_pos = (1,1,1,1), knot_val = (1.,1.,1.,1.), rescale_gamma = False, rescale_amp = 1., rescale_slope = -0.7, seed = 9281110, redshift=99, redend = 2, omegac=0.2408, omegab=0.0472, hubble=0.7, scalar_amp=2.427e-9, ns=0.97, uvb="hm"):
        #Parameters of the heating rate rescaling to account for helium reionisation
        #Default parameters do nothing
        self.rescale_gamma = rescale_gamma
        self.rescale_amp = rescale_amp
        self.rescale_slope = rescale_slope
        #Set up the knot parameters
        self.knot_pos = knot_pos
        self.knot_val = knot_val
        knot_names = list(string.ascii_lowercase[:len(knot_pos)])
        #Set up new output directory hierarchy for Lyman alpha simulations
        #First give number of knots and their positions
        knot_spec = [kn+str(kp) for (kn, kp) in zip(knot_names, self.knot_pos)]
        new_outdir = os.path.join(outdir, "".join(self.knot_spec))
        #Then box and npart, as we will want to correct by these
        new_outdir = os.path.join(os.path.join(new_outdir, str(box)), str(npart))
        #Then the knot values that have changed - we may want to add thermal parameters or cosmology here at some point
        knot_changed = [kn+str(kv) for (kn,kv) in zip(knot_names, self.knot_val) if kv != 1.]
        new_outdir = os.path.join(new_outdir,"knot_"+"".join(self.knot_changed))
        #Make this directory tree
        os.makedirs(new_outdir)
        simulation.Simulation.__init__(self, outdir=new_outdir, box=box, npart=npart, seed=seed, redshift=redshift, redend=redend, separate_gas=True, omegac=omegac, omegab=omegab, hubble=hubble, scalar_amp=scalar_amp, ns=ns, uvb=uvb)
        self.camb_times = [9,]+[x for x in np.arange(4.2,1.9,-0.2)]

    def _feedback_config_options(self, config):
        """Config options specific to the Lyman alpha forest star formation criterion"""
        config.write("SFR\n")
        config.write("QUICK_LYALPHA\n")
        if self.rescale_gamma:
            config.write("RESCALE_EEOS\n")
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

    def change_power_spectrum_knots(self):
        """Multiply the power spectrum file by a function specified by our four knots.
        We assume that the power spectrum is linearly interpolated between the knots,
        so that we preserve additivity:
        ie, P(k | A =1.1, B=1.1) / P(k | A =1, B=1) == P(k | A =1.1) / P(k | A =1.)+ P(k | B =1.1) / P(k | A =B.)"""
        #How to handle interpolation?
        #TODO: THIS NEEDS A TEST

if __name__ == "__main__":
    LymanAlphaSim = simulation.coma_mpi_decorate(LymanAlphaSim)
    ss = LymanAlphaSim(outdir=os.path.expanduser("~/data/Lya_Boss/test1"), box=60, npart=512)
    ss.make_simulation()
