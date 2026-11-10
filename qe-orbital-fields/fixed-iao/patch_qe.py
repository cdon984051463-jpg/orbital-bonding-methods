# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Install the three fixed-projector hooks into a separate QE 7.5 source copy."""
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:])
edits = {
    'h_psi': [
        ('  USE fft_helper_subroutines\n', '  USE fft_helper_subroutines\n  USE iao_fixed_field, ONLY: iao_hpsi\n'),
        ('  IF ( gamma_only .AND. gstart == 2 ) &',
         '  CALL iao_hpsi(lda,n,m,psi,hpsi)\n  IF ( gamma_only .AND. gstart == 2 ) &'),
    ],
    'run_pwscf': [
        ('  USE kinds,                ONLY : DP\n',
         '  USE kinds,                ONLY : DP\n  USE iao_fixed_field, ONLY: iao_initialize\n'),
        ('\n  CALL init_run()\n  !\n  !  read external force fields parameters\n',
         '\n  CALL iao_initialize()\n  CALL init_run()\n  !\n  !  read external force fields parameters\n'),
    ],
    'forces': [
        ('  USE kinds,             ONLY : DP\n',
         '  USE kinds,             ONLY : DP\n  USE iao_fixed_field, ONLY: iao_observables, iao_raw_force\n'),
        ('  ! ... on the crystal\n  !\n  DO ipol = 1, 3\n',
         '  ! ... on the crystal\n  !\n  CALL iao_observables()\n  DO ipol = 1, 3\n'),
        ('     !TB\n     IF ((gate.AND.relaxz).AND.(ipol==3))',
         '     CALL iao_raw_force(ipol,force(ipol,:))\n     !TB\n     IF ((gate.AND.relaxz).AND.(ipol==3))'),
    ],
}
output.mkdir(parents=True, exist_ok=False)
for name, replacements in edits.items():
    text = (source / 'PW/src' / (name + '.f90')).read_text()
    for anchor, replacement in replacements:
        if text.count(anchor) != 1:
            raise ValueError('Expected one pristine QE 7.5 anchor in ' + name)
        text = text.replace(anchor, replacement)
    (output / (name + '.f90')).write_text(text, newline='\n')
