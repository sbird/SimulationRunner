"""Specialization of the Simulation class to Lyman-alpha forest simulations."""

import os
import string
import numpy as np
import scipy.interpolate as interp
import simulation
import clusters

class LymanAlphaSim(simulation.Simulation):
    """Specialise the Simulation class for the Lyman alpha forest.
        This uses the QuickLya star formation module and allows for altering the power spectrum with knots
        """
    __doc__ = __doc__+simulation.Simulation.__doc__
    def __init__(self, outdir, box, npart, *, knot_pos = (0.15,0.475,0.75,1.19), knot_val = (1.,1.,1.,1.), rescale_gamma = False, rescale_amp = 1., rescale_slope = -0.7, seed = 9281110, redshift=99, redend = 2, omegac=0.2408, omegab=0.0472, hubble=0.7, scalar_amp=2.427e-9, ns=0.97, uvb="hm"):
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
        new_outdir = os.path.join(outdir, "".join(knot_spec))
        #Then box and npart, as we will want to correct by these
        new_outdir = os.path.join(os.path.join(new_outdir, str(box)), str(npart))
        #Then the knot values that have changed - we may want to add thermal parameters or cosmology here at some point
        knot_changed = [kn+str(kv) for (kn,kv) in zip(knot_names, self.knot_val) if kv != 1.]
        new_outdir = os.path.join(new_outdir,"knot_"+"".join(knot_changed))
        #Make this directory tree
        try:
            os.makedirs(new_outdir)
        except FileExistsError:
            pass
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

    def _alter_power(self, camb_output):
        """Generate a new CAMB power spectrum multiplied by the knot values."""
        camb_file = camb_output+"_matterpow_"+str(self.redshift)+".dat"
        matpow = np.loadtxt(camb_file)
        matpow2 = change_power_spectrum_knots(self.knot_pos, self.knot_val, matpow)
        #Save a copy of the old file
        os.rename(camb_file, camb_file+".orig")
        np.savetxt(camb_file, matpow2)
        return

def change_power_spectrum_knots(knotpos, knotval, matpow):
    """Multiply the power spectrum file by a function specified by our knots.
    We assume that the power spectrum is linearly interpolated between the knots,
    so that we preserve additivity:
    ie, P(k | A =1.1, B=1.1) / P(k | A =1, B=1) == P(k | A =1.1) / P(k | A =1.)+ P(k | B =1.1) / P(k | A =B.)
    On scales larger and smaller than the specified knots, the power spectrum is changed by the same factor as the last knot specified.
    So if the smallest knotval is 1.1, P(k) from k = 0 -> knotpos[0] is multiplied by 1.1.
    Note that this means that if you want the large scales to be unchanged, you should impose an extra, fixed, knot that stays constant."""
    #This should catch some cases where we pass the arguments in the wrong order
    assert np.all([k1 < k1p for (k1, k1p) in zip(knotpos[:-1], knotpos[1:])])
    assert np.shape(knotval) == np.shape(knotpos)
    #Split and copy the matter power spectrum
    kval = np.array(matpow[:,0])
    pval = np.array(matpow[:,1])
    #Check that the input makes physical sense
    assert np.all(knotpos) > 0
    assert np.all(knotpos) > kval[0] and np.all(knotpos) < kval[-1]
    assert np.all(knotval) > 0
    #BOUNDARY CONDITIONS
    #Add knots at the start and end of the matter power spectrum.
    #The large scale knot is always 1.
    #The small-scale knot always follows the last real knot
    ext_knotpos = np.concatenate([[kval[0]*0.95,],knotpos, [kval[-1]*1.05,] ])
    ext_knotval = np.concatenate([[knotval[0],],knotval, [knotval[-1],] ])
    assert np.shape(ext_knotpos) == np.shape(ext_knotval) and np.shape(ext_knotpos) == (np.size(knotval)+2,)
    #Insert extra power spectrum evaluations at each knot, to make sure we capture those points properly.
    #Build an interpolator (in log space) to get new Pk values. Only interpolate a subset of Pk for speed
    i_limits = np.searchsorted(kval, [knotpos[0]*0.66, knotpos[-1]*1.5])
    (imin, imax) = (np.max([0,i_limits[0]-5]), np.min([len(kval)-1,i_limits[-1]+5]))
    pint = interp.interp1d(np.log(kval[imin:imax]), np.log(pval[imin:imax]), kind='cubic')
    #Make sure that points to be inserted are not too close to the already existing ones.
    #avg_distance = np.mean(np.log(kval[imin+1:imax+1]) - np.log(kval[imin:imax]))
    #closest = np.array([np.min(np.abs(kn - kval[imin:imax])) for kn in knotpos])
    #ii = np.where(closest > avg_distance/4)
    #Also add an extra point in the midpoint of their interval. This helps spline interpolators.
    locations = np.searchsorted(kval[imin:imax], knotpos)
    midpoints = (kval[imin:imax][locations] + kval[imin:imax][locations-1])/2.
    kplocs = np.searchsorted(knotpos, midpoints)
    ins_knotpos = np.insert(knotpos, kplocs, midpoints)
    #Now actually add the new points to the Pk array
    index = np.searchsorted(kval, ins_knotpos)
    kval = np.insert(kval, index, ins_knotpos)
    pval = np.insert(pval, index, np.exp(pint(np.log(ins_knotpos))))
    #Linearly interpolate between these values
    dpk = interp.interp1d(ext_knotpos, ext_knotval, kind='linear')
    #Multiply by the knotted power spectrum interpolated to the point given in the power spectrum file.
    pval *= dpk(kval)
    #Build something like the original matpow
    return np.vstack([kval, pval]).T

if __name__ == "__main__":
    LymanAlphaSim = clusters.coma_mpi_decorate(LymanAlphaSim)
    ss = LymanAlphaSim(knot_val = (1.,1.2,1.,1.),outdir=os.path.expanduser("~/data/Lya_Boss/test1"), box=60, npart=512)
    ss.make_simulation()
