# Upstream software and attribution

- Quantum ESPRESSO 7.5: GPL-2.0-or-later. This repository distributes the custom
  Fortran modules and hook definitions, not a complete QE distribution. Obtain
  matching pristine sources and a compatible build from the official project:
  https://www.quantum-espresso.org/
- LOBSTER 5.1.1: obtain the program under its own license from
  https://www.cohp.de/. Only this project's analysis routines and numerical
  example results are included here.
- VASP: obtain the engine and PAW datasets through your own licensed access.
  Numerical example arrays are included; POTCAR and VASP binaries are not.
- Three.js revision 166 and OrbitControls: MIT license. The exact viewer
  library sources used by the existing viewer are included under
  `iao-svd/viewer/` with their license; https://github.com/mrdoob/three.js .
- PyVaspWfc / VaspBandUnfolding by Qijing Zheng:
  https://github.com/QijingZheng/VaspBandUnfolding , revision
  3f8a6285b538cdec1d896e803e3d0800147fda88. Used as an external dependency by
  the PAW preprocessing and field scripts; its source is not copied into this repository.

Core mathematical antecedents to cite as appropriate include intrinsic atomic
orbitals (Knizia, https://doi.org/10.1021/ct400687b), the Quantum ESPRESSO
method papers (https://doi.org/10.1088/0953-8984/21/39/395502 and
https://doi.org/10.1088/1361-648X/aa8f79), and LOBSTER
(Maintz et al., https://doi.org/10.1002/jcc.24300).
These references acknowledge foundations and software; they do not establish
priority for the additional analysis choices in this repository.
