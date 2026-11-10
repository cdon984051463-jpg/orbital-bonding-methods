#!/bin/bash
# QE 7.5 pristine libraries are read only. Output must be a new directory.
set -euo pipefail
base=$1
source=$2
output=$3
here=$(cd "$(dirname "$0")" && pwd)
mkdir "$output"
mkdir "$output/mod" "$output/obj"
python3 "$here/patch_qe.py" "$source" "$output/patched"
flags="$base/PW/CMakeFiles/qe_pw.dir/flags.make"
read -r -a definitions <<< "$(sed -n 's/^Fortran_DEFINES = //p' "$flags")"
read -r -a includes <<< "$(sed -n 's/^Fortran_INCLUDES = //p' "$flags")"
mpif90 "${definitions[@]}" "${includes[@]}" -J"$output/mod" -O2 -g -fcheck=bounds \
  -ffree-line-length-none -fallow-argument-mismatch -cpp -c "$here/iao_fixed_field.f90" \
  -o "$output/obj/iao_fixed_field.o"
for name in h_psi run_pwscf forces; do
  mpif90 "${definitions[@]}" -I"$output/mod" "${includes[@]}" -J"$output/mod" \
    -O2 -g -fcheck=bounds -fallow-argument-mismatch -cpp \
    -c "$output/patched/$name.f90" -o "$output/obj/$name.o"
done
mpif90 -o "$output/pw_iao_fixed.x" "$output"/obj/*.o \
  "$base/PW/CMakeFiles/qe_pw_exe.dir/src/pwscf.f90.o" \
  -Wl,--start-group "$base"/lib/libqe*.a "$base/lib/libmbd.a" -Wl,--end-group \
  -lfftw3 -lopenblas -lm -ldl
mpif90 "${definitions[@]}" "${includes[@]}" -J"$output/mod" -O2 -g -fcheck=bounds \
  -ffree-line-length-none -fallow-argument-mismatch -cpp -c "$here/qe_iao_projector.f90" \
  -o "$output/obj/qe_iao_projector.o"
mpif90 "${definitions[@]}" "${includes[@]}" -J"$output/mod" -O2 -cpp \
  -c "$source/PP/src/stop_pp.f90" -o "$output/obj/stop_pp.o"
mpif90 -o "$output/qe_iao_projector.x" "$output/obj/qe_iao_projector.o" "$output/obj/stop_pp.o" \
  -Wl,--start-group "$base"/lib/libqe*.a "$base/lib/libmbd.a" -Wl,--end-group \
  -lfftw3 -lopenblas -lm -ldl
mpif90 "${definitions[@]}" "${includes[@]}" -J"$output/mod" -O2 -g -fcheck=bounds \
  -ffree-line-length-none -fallow-argument-mismatch -cpp -c "$here/native_backend/qe_iao_export.f90" \
  -o "$output/obj/qe_iao_export.o"
mpif90 -o "$output/qe_iao_export.x" "$output/obj/qe_iao_export.o" "$output/obj/stop_pp.o" \
  -Wl,--start-group "$base"/lib/libqe*.a "$base/lib/libmbd.a" -Wl,--end-group \
  -lfftw3 -lopenblas -lm -ldl
