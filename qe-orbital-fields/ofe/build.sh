#!/usr/bin/env bash
# Build against matching pristine GNU/MPI QE 7.5 CMake libraries; source is read only.
set -euo pipefail
base=$(realpath "$1")
source=$(realpath "$2")
output=$(realpath -m "$3")
here=$(cd "$(dirname "$0")" && pwd)
mkdir "$output"
mkdir "$output/mod" "$output/obj"
python3 "$here/patch_qe.py" "$source" "$output/patched"
flags="$base/PW/CMakeFiles/qe_pw.dir/flags.make"
read -r -a definitions <<< "$(sed -n 's/^Fortran_DEFINES = //p' "$flags")"
read -r -a includes <<< "$(sed -n 's/^Fortran_INCLUDES = //p' "$flags")"
compile() {
    mpif90 "${definitions[@]}" "${includes[@]}" -I"$output/mod" -J"$output/mod" \
        -O2 -g -fcheck=bounds -fallow-argument-mismatch -cpp -c "$1" -o "$2"
}
compile "$here/src/ofe_math.f90" "$output/obj/ofe_math.o"
compile "$here/src/orbital_field_engine.f90" "$output/obj/orbital_field_engine.o"
for name in h_psi run_pwscf forces; do
    compile "$output/patched/PW/src/$name.f90" "$output/obj/$name.o"
done
mpif90 -o "$output/pw_ofe.x" "$output"/obj/*.o \
    "$base/PW/CMakeFiles/qe_pw_exe.dir/src/pwscf.f90.o" \
    -Wl,--start-group "$base"/lib/libqe*.a "$base/lib/libmbd.a" -Wl,--end-group \
    -lfftw3 -lopenblas -lm -ldl
printf 'Built %s\n' "$output/pw_ofe.x"
