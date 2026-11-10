# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Apply the established OFE CPU hooks to pristine QE 7.5 in a new directory."""
from pathlib import Path
import shutil
import sys

CPU_EDITS = {'PW/src/h_psi.f90': [('  USE fft_helper_subroutines\n', '  USE fft_helper_subroutines\n  USE orbital_field_engine,    ONLY: ofe_hpsi\n'), ('  IF ( gamma_only .AND. gstart == 2 ) &', '  CALL ofe_hpsi( lda, n, m, psi, hpsi )\n  !\n  IF ( gamma_only .AND. gstart == 2 ) &')], 'PW/src/run_pwscf.f90': [('  USE kinds,                ONLY : DP\n', '  USE kinds,                ONLY : DP\n  USE orbital_field_engine, ONLY : ofe_initialize\n'), ('\n  CALL init_run()\n  !\n  !  read external force fields parameters\n', '\n  CALL ofe_initialize()\n  CALL init_run()\n  !\n  !  read external force fields parameters\n')], 'PW/src/forces.f90': [('  USE kinds,             ONLY : DP\n', '  USE kinds,             ONLY : DP\n  USE orbital_field_engine, ONLY : ofe_forces, ofe_record_raw_component, ofe_finish_raw, &\n                                   ofe_record_final\n'), ('  ! ... on the crystal\n  !\n  DO ipol = 1, 3\n', '  ! ... on the crystal\n  !\n  CALL ofe_forces( force )\n  DO ipol = 1, 3\n'), ('     !TB\n     IF ((gate.AND.relaxz).AND.(ipol==3))', '     CALL ofe_record_raw_component( ipol, force(ipol,:) )\n     !TB\n     IF ((gate.AND.relaxz).AND.(ipol==3))'), ('  ! ... call run_extffield to apply external force fields on ions\n', '  CALL ofe_finish_raw()\n  ! ... call run_extffield to apply external force fields on ions\n'), ('  ! ... write on output the forces\n', '  CALL ofe_record_final( force )\n  !\n  ! ... write on output the forces\n')], 'PW/CMakeLists.txt': [('set(src_pw\n', 'set(src_pw\n   src/ofe_math.f90\n   src/orbital_field_engine.f90\n')]}

source, output = [Path(value).resolve() for value in sys.argv[1:]]
if source == output or source in output.parents:
    raise ValueError("Output must be outside the source tree")
patched = {}
for relative, edits in CPU_EDITS.items():
    text = (source / relative).read_text(encoding="utf-8")
    for anchor, replacement in edits:
        if text.count(anchor) != 1:
            raise ValueError(f"QE 7.5 hook must occur exactly once: {relative}: {anchor!r}")
        text = text.replace(anchor, replacement, 1)
    patched[relative] = text
output.mkdir()
for relative, text in patched.items():
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
for name in ("ofe_math.f90", "orbital_field_engine.f90"):
    shutil.copyfile(Path(__file__).parent / "src" / name, output / "PW/src" / name)
print("OFE V1.3 CPU hooks applied to a new source subset")
