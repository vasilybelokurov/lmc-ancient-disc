# The puzzle of the ancient stellar disc of the Large Magellanic Cloud

This site is the knowledge base of the MNRAS Letter *The puzzle of the ancient stellar disc of the Large Magellanic Cloud* (Belokurov,
in preparation). It shows how every figure of the Letter is made, from which data and with which scripts, and it holds the material
that did not fit in the Letter.

**The result in brief.** The LMC's red giants rotate as a disc at every metallicity, including the most metal-poor ones, which
formed when the LMC had made only a few per cent of its stars. Simulated LMC analogues (TNG50, Auriga, FIRE-2) do not keep such an
early disc: their oldest stars are born rotating but lose the rotation within one or two orbital periods, because the young dark
matter halo is aspherical and misaligned with the disc. A self-interacting dark-matter halo is round early, which could let an early
disc survive.

**How to use it.** Each page about a Letter figure is a notebook whose code cells are the steps of the figure script. They remake
the figure from the result tables of the repository when this site is built; the page then shows the figure with the Letter's
caption and its numbers, and lists the steps upstream of the figure, down to the public data. 'Open in Colab' runs the same
notebook cell by cell.

The chapters of *The full analysis* go through all the work, step by step: from the public surveys and simulations to the
catalogues, the selection function, the stellar mass, the kinematic fit, the shape, the simulated analogues, and the tables and
numbers of the text. Steps that need the raw survey catalogues, the simulation particles or hours of computing are shown as code,
marked *Shown, not run here*, with what they need. The scrambling experiment of Fig. 4 can be run for one cohort, on the site and
in Colab.

| Page | Figure |
|---|---|
| [Fig. 1](notebooks/letter/fig1_tracers) | the tracers, the mass they formed, and their rotation |
| [Fig. 2](notebooks/letter/fig2_rotational_support) | rotational support against metallicity, the LMC and its simulated analogues |
| [Fig. 3](notebooks/letter/fig3_birth) | the simulated stars at birth and today |
| [Fig. 4](notebooks/letter/fig4_scrambling) | how the early discs lose their rotation, and why |
| [Fig. 5](notebooks/letter/fig5_sidm) | the centre of an LMC-mass halo in cold and self-interacting dark matter |

The code is at [github.com/vasilybelokurov/lmc-ancient-disc](https://github.com/vasilybelokurov/lmc-ancient-disc).
