"""Short module to contain code to read UVB tables from two different sources:

 Faucher-Giguere 2009:
 ( http://galaxies.northwestern.edu/uvb/ ) at time of writing
"A New Calculation of the Ionizing Background Spectrum
and the Effects of HeII Reionization" (arXiv:0901.4554)
by C.-A. Faucher-Giguere, A. Lidz, M. Zaldarriaga, & L. Hernquist
(December 2011 update)

Format of FG2009 is that needed for gadget and is:
 - TREECOOL_fg_dec11.dat:
log(1+z), Gamma_HI, Gamma_HeI, Gamma_HeII,  Qdot_HI, Qdot_HeI, Qdot_HeII,
where 'Gamma' is the photoionization rate and 'Qdot' is the photoheating rate.
The Gamma's are in units of s^-1, and the Qdot's are in units of erg s^-1.

Haardt Madau 2012:
 ( http://www.ucolick.org/~pmadau/CUBA/HOME.html ) at time of writing.
 Haardt & Madau  (2012, ApJ, 746, 125)

Format is:
**Photoheating rates are in *eV/s* and thus ~6e11 times LARGER than in FG2009.**

 The following 7 lines are column identification labels
 z=redshift, HI_i=hydrogen phoionization rate, HI_h=hydrogen photoheating rate
 HeI_i,h=photoionization and photoheating rates for neutral helium
 HeII_i,h=photoionization and photoheating rates for singly ionized helium
 Compton= Compton heating rate.
 All photoionization and photoheating rates are *per ion* and have units of 1/s and eV/s, respectively.
 The Compton heating rate is *per electron* and has units of eV/s.
  z      HI_i     HI_h    HeI_i    HeI_h    HeII_i   HeII_h   Compton

"""

import numpy as np

def format_HM12_UVB(HM_in_file, HM_out_file):
    """Alter an HM2012 file to have the format required by Gadget. This involves reordering the columns
    and changing the photoheating units."""
    #One erg in eV, to convert the units on the photoheating rates.
    eV_in_erg = 1.60218e-12
    #Read table
    hm_in_table = np.loadtxt(HM_in_file)
    #Check redshifts
    assert np.min(hm_in_table[:,0]) >= 0 and np.max(hm_in_table[:,0]) < 16
    #Reorder photo heat and photoion columns
    photoheat = eV_in_erg*np.array([hm_in_table[:,2], hm_in_table[:,4],hm_in_table[:,6]])
    photoion = np.array([hm_in_table[:,1],hm_in_table[:,3], hm_in_table[:,5]])
    #Make output table
    hm_out_table = np.vstack([hm_in_table[:,0], photoion, photoheat]).T
    #Check shape
    assert np.shape(hm_out_table) == (59,7)
    #Check we did the conversion the right way
    assert np.max(hm_out_table[:,4:]) < 1e-23
    #(Do not check HeII ion rate as it overlaps heating rate)
    assert np.min(hm_out_table[:,1:3]) > 1e-16
    np.savetxt(HM_out_file, hm_out_table, fmt="%1.6e")
    return hm_out_table

def get_fg11_filename():
    """File where the FG2009 table is stored"""
    return "TREECOOL_fg_dec11"

def get_hm12_filename():
    """File where the HM2012 table is stored (in gadget format)"""
    return "TREECOOL_hm_2012"

def get_uvb_filename(uvb):
    """Get the filename for a UVB table (in gadget format)"""
    if uvb == "fg":
        fuvb = get_fg11_filename()
    if uvb == "hm":
        fuvb = get_hm12_filename()
    else:
        raise ValueError("Unsupported UVB table")
    return fuvb
