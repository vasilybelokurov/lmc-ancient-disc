"""Reference parameters of the LMC and the SMC: the only place in the project where these numbers are written.

Angles in degrees, distances in kpc, proper motions in mas/yr (mu_alpha* = mu_alpha cos(delta)), velocities in km/s. Every value
quotes its source. No other module or script may hard-code any of them. A value added or changed here must carry its source
(paper, table or column, bibcode or URL) in a comment next to it.

LMC: the adopted reference values. For now they are van der Marel & Kallivayalil (2014, ApJ 781, 121; arXiv:1305.4641), Table 1,
column (4) "PMs+Old vLOS"; they will be replaced by this paper's own fit, and this docstring will then say so. Conventions (their
Fig. 2): theta is the position angle of the line of nodes (N through E); the near side is at theta - 90 deg; their mu_W = -mu_alpha*.

Besides the reference parameters, this module holds the published values that the figures and tables compare with (stellar
masses, [Fe/H] gradients and levels, the age-metallicity relation), each with its bibcode or reference. Bibkeys in
LMC_MASS_LITERATURE refer to the Letter's bibliography (paper/refs.bib); the matching sources.yaml entries are
lmc_mass_reference_<key> (fetched by scripts/fetch_lmc_mass_literature.py).
"""

# km/s per (mas/yr x kpc): 1 au per Julian year (IAU 2012 au; Gaia uses Julian years). Tangential velocity
# v_t [km/s] = KAPPA x mu [mas/yr] x D [kpc].
KAPPA = 149597870.7 / 31557600.0  # = 4.740470463533348
# km/s per kpc/Gyr: 1 kpc = 3.0856775814913673e16 km (IAU 2012 au, parsec = 648000/pi au), 1 Gyr = 1e9 Julian years.
KPC_GYR = 3.0856775814913673e16 / (1e9 * 31557600.0)  # = 0.9777922216807891
# Gravitational constant in kpc (km/s)^2 / Msun (CODATA 2018 G and IAU 2015 nominal solar mass parameter, as in astropy 6).
G = 4.300917270036279e-06

# LMC reference parameters (vdMK14 Table 1, column (4), unless stated); uncertainties quoted in the comments are theirs.
LMC = dict(
    ra0=79.88,     # deg, centre
    dec0=-69.59,   # deg
    inc=34.0,      # deg, +/- 7.0
    theta=139.1,   # deg, +/- 4.1
    D0=50.1,       # kpc (distance modulus 18.50, Freedman et al. 2001, as adopted by vdMK14)
    pmra0=1.895,   # mas/yr, systemic (mu_W = -1.895 +/- 0.024)
    pmdec0=0.287,  # mas/yr, +/- 0.054
    vlos0=261.1,   # km/s, +/- 2.2
    v0=83.8,       # km/s, rotation curve amplitude (V0,PM, +/- 9.0): rising linearly to R0, flat beyond
    R0=3.76,       # kpc, turnover radius (R0/D0 = 0.075)
)

# SMC centre and systemic proper motion: the RR Lyrae membership mask (lmcold.footprint) and the membership and map figures.
SMC = dict(
    ra0=13.187,    # deg, HI kinematic centre (Stanimirovic et al. 2004, 00h52m44.8s -72d49m43s), as used by Zivick et al. 2018.
    dec0=-72.829,  # UNVERIFIED against Stanimirovic et al. 2004; only used to place the SMC mask, whose edge is 13 deg away.
    pmra0=0.82,    # mas/yr, Zivick et al. 2018 (ApJ 864, 55): mu_W = -0.82 +/- 0.02 (random) +/- 0.10 (systematic)
    pmdec0=-1.21,  # mas/yr, mu_N = -1.21 +/- 0.01 (random) +/- 0.03 (systematic)
)

# Photometric distance for stellar-population models (separate from the reference
# kinematic D0 above): Pietrzynski et al. 2019, Nature 567, 200, arXiv:1903.08096.
LMC_DISTANCE_MODULUS = 18.477
LMC_DEPTH_MAG = 0.1  # mag, adopted Gaussian line-of-sight width; as in lmcold.isochrones.count_grid

# Extinction-mask centres in the Magellanic Stream frame: SDSS target_selection
# mwm_magcloud_rgb carton, plan 1.0.50 (targeting geometry, not fitted centres).
# (L, B) in degrees, Nidever et al. 2008 frame; lmcold.magcloud rescales SFD E(B-V) within 4.5 deg (LMC) and 2.5 deg (SMC)
# of these points.
LMC_TARGET_MS = (-0.6, 3.6)  # LMC (L, B) [deg], mwm_magcloud_rgb carton, plan 1.0.50
SMC_TARGET_MS = (-15.53, -11.58)  # SMC (L, B) [deg], same carton source as LMC_TARGET_MS (plan 1.0.50)

# Present-day stellar masses for the external RR census comparison (Msun; scripts/paper_rr_mass_comparison.py).
# Skibba et al. 2012, ApJ 761, 42, section 3.2.2 (Salpeter calibration).
MAGELLANIC_STELLAR_MASS_SKIBBA12 = {'LMC': 2.0e9, 'SMC': 3.1e8}
# van der Marel 2004 review, section 5.2, quoting van der Marel et al. 2002.
# Alternative literature normalization, not an uncertainty interval.
LMC_STELLAR_MASS_VDM = 2.7e9

# Published LMC stellar-mass benchmarks. Numbers retain original IMF scales;
# these are comparisons, not parameters adopted for our mass calculation.
# HZ09 is approximate from Fig.12 and explicitly quoted by Mazzi2021 section5.5.
# Fields: key (sources.yaml suffix), bibkey (refs.bib), summary/imf_short/label (LaTeX text for Table 3 and the figure),
# kind ('formed' or 'present' mass), mass [Msun] as published, imf/coverage/method/location (provenance notes),
# family (estimates sharing data or calibration), approx_kroupa (optional illustrative Salpeter -> Kroupa factor, None if not
# applied), retained_fraction (present/formed fraction the authors applied), source_key (sources.yaml entry quoting the value,
# when it is not the paper itself). Read by scripts/paper_mass_summary.py and scripts/paper_table_mass_summary.py.
LMC_MASS_LITERATURE = [
    dict(key='kim1998',bibkey='Kim1998',summary=r'H\,{\sc i} rotation curve, inner 4\,kpc',imf_short=r'--',label='Kim et al. (1998)',kind='present',
         mass=2.0e9,imf='Rotation-curve decomposition; no population IMF specified',
         coverage='Within 4 kpc; smaller than the full LMC',method='Stellar component of rotation-curve model',
         location='1998ApJ...503..674K; stellar value explicitly quoted in Keim2023 introduction',
         family='HI rotation curve',approx_kroupa=None,source_key='keim2023'),
    dict(key='vanderMarel2002',bibkey='vanderMarel2002',summary=r'integrated $V$ light, $M/L_V\approx0.9$',imf_short=r'colour $M/L$',label='van der Marel et al. (2002)',kind='present',
         mass=LMC_STELLAR_MASS_VDM,imf='Bell & de Jong colour-M/L; no unique IMF conversion adopted',
         coverage='Total stellar disk from integrated V luminosity',method='Colour-dependent M/L_V ~0.9',
         location='Section 8.4, stellar disk versus dynamical mass',family='integrated optical',approx_kroupa=None),
    dict(key='harris_zaritsky2009',bibkey='Harris2009',summary=r'optical CMD history, 64\,deg$^2$',imf_short=r'Salpeter',label='Harris & Zaritsky (2009)',kind='formed',
         mass=2.2e9,imf='Salpeter; normalization details insufficient for a uniform conversion',
         coverage='MCPS main-body footprint, ~64 square degrees',method='Resolved optical CMD star formation history',
         location='Section 5.1 and Fig.12; approximate total quoted by Mazzi2021 section5.5',family='MCPS SFH',approx_kroupa=None),
    dict(key='mcconnachie2012',bibkey='McConnachie2012',summary=r'integrated light, $M/L_V=1$',imf_short=r'--',label='McConnachie (2012)',kind='present',
         mass=1.5e9,imf='Not specified: assumed M/L_V=1',
         coverage='Integrated luminosity compilation',method='Fixed stellar M/L_V=1; not a population fit',
         location='Section 5 and Table4',family='integrated optical',approx_kroupa=None),
    dict(key='eskew2012',bibkey='Eskew2012',summary=r'infrared flux on the Harris \& Zaritsky history',imf_short=r'Salpeter',label='Eskew et al. (2012)',kind='present',
         mass=1.87e9,imf='Salpeter; authors apply 0.94 retained fraction',
         coverage='1180 MCPS regions matched to SAGE; all regions, SFH-derived mass',
         method='HZ09 SFH mass calibrates infrared flux',location='Section3, all-region SFH total after returned-mass correction',
         family='MCPS SFH',approx_kroupa=.66,retained_fraction=.94),
    dict(key='skibba2012',bibkey='Skibba2012',summary=r'infrared maps, Eskew et al. calibration',imf_short=r'Salpeter',label='Skibba et al. (2012)',kind='present',
         mass=MAGELLANIC_STELLAR_MASS_SKIBBA12['LMC'],imf='Salpeter via Eskew2012',
         coverage='Integrated stellar surface-density maps',method='Infrared mass using Eskew2012 calibration',
         location='Section3.2.2',family='MCPS SFH',approx_kroupa=.66,retained_fraction=.94),
    dict(key='mazzi2021',bibkey='Mazzi2021',summary=r'near-infrared CMD history, 96\,deg$^2$',imf_short=r'Kroupa',label='Mazzi et al. (2021)',kind='formed',
         mass=2.5e9,imf='Kroupa 0.01–250 Msun, including binaries',
         coverage='VMC main body, 96 square degrees',method='Resolved near-infrared CMD star formation history',
         location='Sections3.4,5.3,5.5 and Fig.17',family='VMC SFH',approx_kroupa=None),
]
# The optional .66 illustrative rescaling above is the stellar-mass factor in
# Madau & Dickinson2014 (ARA&A52,415), not an exact recalibration of MCPS data.

# Published LMC radial [Fe/H] gradients [dex/kpc] over in-plane radius r_lo-r_hi [kpc], from the ADS abstracts; every value
# was also found in the published full text by a SciX full-text query (2026-10-27; Povick+25: the arXiv preprint's -0.0380 is
# not in the published version). range_from_deg: the abstract gives the extent in degrees, converted here at 0.873 kpc/deg (D0 = 50.1 kpc,
# projected), so the range is approximate. slope_hi: upper end of a quoted interval of slopes. err None: none quoted.
# Read by scripts/paper_feh_profile.py (Fig. 5a, b) and scripts/paper_feh_literature.py (appendix E).
LMC_FEH_GRADIENT_LITERATURE = [
    dict(label='Povick+25', bibcode='2025MNRAS.544..457P', tracer='giants', method='APOGEE spectra',
         slope=-0.0296, err=0.0017, r_lo=0., r_hi=8.7, range_from_deg=True),
    dict(label='Grady+21', bibcode='2021ApJ...909..150G', tracer='giants', method='Gaia DR2 photometric',
         slope=-0.048, err=0.001, r_lo=0., r_hi=10.5, range_from_deg=True),
    dict(label='Cioni 09', bibcode='2009A&A...506.1137C', tracer='giants', method='AGB C/M ratio',
         slope=-0.047, err=0.003, r_lo=0., r_hi=8., range_from_deg=False),
    dict(label='Choudhury+16', bibcode='2016MNRAS.455.1855C', tracer='giants', method='RGB slope, optical',
         slope=-0.049, slope_hi=-0.066, err=0.006, r_lo=0., r_hi=4., range_from_deg=False),
    dict(label='Choudhury+21', bibcode='2021MNRAS.507.4752C', tracer='giants', method='RGB slope, near-infrared',
         slope=-0.008, err=0.001, r_lo=0., r_hi=6., range_from_deg=False),
    dict(label='Skowron+16 inner', bibcode='2016AcA....66..269S', tracer='RRab', method='OGLE-IV I-band phi31',
         slope=-0.029, err=0.002, r_lo=0., r_hi=5., range_from_deg=False),
    dict(label='Skowron+16 middle', bibcode='2016AcA....66..269S', tracer='RRab', method='OGLE-IV I-band phi31',
         slope=0., err=None, r_lo=5., r_hi=8., range_from_deg=False),
    dict(label='Skowron+16 outer', bibcode='2016AcA....66..269S', tracer='RRab', method='OGLE-IV I-band phi31',
         slope=-0.030, err=0.003, r_lo=8., r_hi=10., range_from_deg=False),  # "beyond 8 kpc": outer edge set by our 10-kpc limit
]
# Gradient quoted per degree of projected distance: Munoz et al. 2023 (APOGEE DR17, medians of LMC main-body regions, 0-11 deg;
# full text: "a slope of -0.03 dex deg-1"; outer 6-11 deg "about 0.1 dex more metal poor" than the inner < 6 deg).
LMC_FEH_GRADIENT_DEG_LITERATURE = [
    dict(label='Muñoz+23', bibcode='2023A&A...680A..79M', slope=-0.03, rho_lo=0., rho_hi=11.),
]
# Mean-[Fe/H] levels quoted against projected distance [deg]: Carrera et al. 2008 (2008AJ....135..836C; Ca triplet, fields
# north of the bar: ~-0.5 to 6 deg, ~-0.8 at 8 deg) and Massana et al. 2024 (2024MNRAS.527.8706M; Gaia XP [M/H]: ~-0.8 at
# 3-7 deg, ~-1.0 at 10-18 deg). Fields: rho_lo, rho_hi projected distance range [deg], feh [dex]; read by
# scripts/paper_feh_literature.py.
LMC_FEH_LEVEL_LITERATURE = [
    dict(label='Carrera+08', rho_lo=3., rho_hi=6., feh=-0.5),
    dict(label='Carrera+08', rho_lo=7.7, rho_hi=8.3, feh=-0.8),
    dict(label='Massana+24', rho_lo=3., rho_hi=7., feh=-0.8),
    dict(label='Massana+24', rho_lo=10., rho_hi=12., feh=-1.0),
]

# LMC age-metallicity relation of the field red giants: median [Fe/H] (APOGEE scale) and 1-sigma scatter against age [Gyr], read by
# eye (+-0.05 dex) from the dashed line and grey band of Fig. 20 of Povick et al. 2024 (MNRAS 533, 3685; 2024MNRAS.533.3685P), from
# ~6000 APOGEE giants. Their ages above 12.5 Gyr (to 20 Gyr) are not used: the PARSEC grid ends at 12.59 Gyr.
# Used as the 'AMR' age weighting of the mass conversion (lmcold.mass_conversion.age_weights): age [Gyr], feh [dex], sigma [dex].
LMC_AMR_POVICK24 = dict(age=[0., 2.5, 5., 7.5, 10., 12.5], feh=[-0.40, -0.57, -0.70, -0.83, -0.92, -0.95],
                        sigma=[0.15, 0.15, 0.18, 0.22, 0.23, 0.24])
