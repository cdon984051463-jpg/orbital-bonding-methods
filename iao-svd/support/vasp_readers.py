"""Support routines retained from the established project implementation."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
import numpy as np

HSQDTM = 3.810019854  # hbar^2 / (2 m_e), eV Angstrom^2


_COEFFICIENT_DTYPES = {
    45200: np.dtype(np.complex64),
    45210: np.dtype(np.complex128),
}


@dataclass(frozen=True)
class Site:
    index: int
    element: str
    cartesian: tuple[float, float, float]
    fractional: tuple[float, float, float]


@dataclass(frozen=True)
class Structure:
    title: str
    lattice: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    sites: list[Site]


def _parse_vector(line: str) -> tuple[float, float, float]:
    parts = line.split()
    if len(parts) < 3:
        raise ValueError(f"Expected three coordinates, got: {line!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _mat_vec_mul(
    vec: tuple[float, float, float],
    lattice: tuple[tuple[float, float, float], ...],
) -> tuple[float, float, float]:
    return (
        vec[0] * lattice[0][0] + vec[1] * lattice[1][0] + vec[2] * lattice[2][0],
        vec[0] * lattice[0][1] + vec[1] * lattice[1][1] + vec[2] * lattice[2][1],
        vec[0] * lattice[0][2] + vec[1] * lattice[1][2] + vec[2] * lattice[2][2],
    )


def _inverse_3x3(
    matrix: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    a, b, c = matrix
    determinant = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    if abs(determinant) < 1e-14:
        raise ValueError("Singular lattice matrix in POSCAR")
    return (
        (
            (b[1] * c[2] - b[2] * c[1]) / determinant,
            (a[2] * c[1] - a[1] * c[2]) / determinant,
            (a[1] * b[2] - a[2] * b[1]) / determinant,
        ),
        (
            (b[2] * c[0] - b[0] * c[2]) / determinant,
            (a[0] * c[2] - a[2] * c[0]) / determinant,
            (a[2] * b[0] - a[0] * b[2]) / determinant,
        ),
        (
            (b[0] * c[1] - b[1] * c[0]) / determinant,
            (a[1] * c[0] - a[0] * c[1]) / determinant,
            (a[0] * b[1] - a[1] * b[0]) / determinant,
        ),
    )


def _cart_to_frac(
    cart: tuple[float, float, float],
    lattice: tuple[tuple[float, float, float], ...],
) -> tuple[float, float, float]:
    return _mat_vec_mul(cart, _inverse_3x3(lattice))


def parse_poscar(path: str | Path) -> Structure:
    """Parse the VASP-5 POSCAR subset used by the method.

    The coordinate and scale behavior is kept numerically identical to the
    previously used reader for positive scalar scale factors.
    """

    source = Path(path)
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 8:
        raise ValueError(f"POSCAR is too short: {source}")
    title = lines[0].strip()
    scale = float(lines[1].split()[0])
    lattice = tuple(
        tuple(scale * value for value in _parse_vector(lines[index]))
        for index in range(2, 5)
    )
    species = lines[5].split()
    try:
        counts = [int(value) for value in lines[6].split()]
    except ValueError as error:
        raise ValueError(f"Invalid species counts in {source}") from error
    if len(species) != len(counts):
        raise ValueError(f"Species/count mismatch in {source}")
    if any(count < 0 for count in counts):
        raise ValueError(f"Negative species count in {source}")

    coordinate_mode = lines[7].strip().lower()
    coordinate_start = 8
    if coordinate_mode.startswith("s"):
        if len(lines) < 9:
            raise ValueError(f"Missing coordinate mode after Selective dynamics: {source}")
        coordinate_mode = lines[8].strip().lower()
        coordinate_start = 9
    if not (coordinate_mode.startswith("d") or coordinate_mode.startswith(("c", "k"))):
        raise ValueError(f"Unrecognized POSCAR coordinate mode: {coordinate_mode!r}")
    is_direct = coordinate_mode.startswith("d")

    expanded_species = [
        element for element, count in zip(species, counts) for _ in range(count)
    ]
    if len(lines) < coordinate_start + len(expanded_species):
        raise ValueError(
            f"POSCAR coordinate block is truncated: expected {len(expanded_species)} sites"
        )
    sites: list[Site] = []
    for offset, element in enumerate(expanded_species):
        vector = _parse_vector(lines[coordinate_start + offset])
        if is_direct:
            fractional = vector
            cartesian = _mat_vec_mul(fractional, lattice)
        else:
            cartesian = tuple(scale * value for value in vector)
            fractional = _cart_to_frac(cartesian, lattice)
        sites.append(
            Site(
                index=offset + 1,
                element=element,
                cartesian=cartesian,
                fractional=fractional,
            )
        )
    return Structure(title=title, lattice=lattice, sites=sites)


class Wavecar:
    """Read the standard VASP WAVECAR records required by this workflow."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.f: BinaryIO | None = None
        self._gcache: dict[int, np.ndarray] = {}
        try:
            self.f = self.path.open("rb")
            header = self._read_array(np.float64, 3, 0, "first WAVECAR header")
            self.recl = self._checked_integer(header[0], "RECL", minimum=24)
            self.nspin = self._checked_integer(header[1], "NSPIN", minimum=1)
            self.rtag = self._checked_integer(header[2], "RTAG", minimum=1)
            if self.rtag not in _COEFFICIENT_DTYPES:
                raise ValueError(
                    f"Unsupported WAVECAR RTAG {self.rtag}; "
                    f"expected one of {sorted(_COEFFICIENT_DTYPES)}"
                )
            self.coefficient_dtype = _COEFFICIENT_DTYPES[self.rtag]
            if self.recl % 8:
                raise ValueError(f"Invalid WAVECAR RECL {self.recl}: not divisible by 8")

            second = self._read_array(
                np.float64, 12, self.recl, "second WAVECAR header"
            )
            self.nk = self._checked_integer(second[0], "NKPOINTS", minimum=1)
            self.nb = self._checked_integer(second[1], "NBANDS", minimum=1)
            self.encut = float(second[2])
            if not np.isfinite(self.encut) or self.encut <= 0:
                raise ValueError(f"Invalid WAVECAR ENCUT {self.encut!r}")
            self.A = np.asarray(second[3:12], dtype=np.float64).reshape(3, 3)
            if not np.all(np.isfinite(self.A)):
                raise ValueError("WAVECAR lattice contains non-finite values")
            determinant = float(np.linalg.det(self.A))
            if abs(determinant) < 1e-14:
                raise ValueError("Singular lattice in WAVECAR")
            self.B = np.linalg.inv(self.A).T
            self.vol = determinant

            minimum_header_bytes = (4 + 3 * self.nb) * np.dtype(np.float64).itemsize
            if minimum_header_bytes > self.recl:
                raise ValueError(
                    f"WAVECAR RECL {self.recl} is too short for {self.nb} bands"
                )
            self.krec: dict[int, int] = {}
            self.kinfo: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
            record = 2
            for ik in range(self.nk):
                values = self._read_array(
                    np.float64,
                    4 + 3 * self.nb,
                    record * self.recl,
                    f"k-point header {ik + 1}",
                )
                nplane = self._checked_integer(
                    values[0], f"k-point {ik + 1} plane-wave count", minimum=1
                )
                if nplane * self.coefficient_dtype.itemsize > self.recl:
                    raise ValueError(
                        f"k-point {ik + 1}: {nplane} coefficients do not fit "
                        f"WAVECAR record length {self.recl}"
                    )
                kvector = np.asarray(values[1:4], dtype=np.float64)
                eigenvalues = np.asarray(values[4::3][: self.nb], dtype=np.float64)
                occupations = np.asarray(values[6::3][: self.nb], dtype=np.float64)
                if not (
                    np.all(np.isfinite(kvector))
                    and np.all(np.isfinite(eigenvalues))
                    and np.all(np.isfinite(occupations))
                ):
                    raise ValueError(f"k-point {ik + 1} header contains non-finite data")
                self.kinfo.append((nplane, kvector, eigenvalues, occupations))
                self.krec[ik] = record
                record += 1 + self.nb
            self._validate_file_size(record)
            self.ng = self._minimum_fft_grid()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _checked_integer(value: float, label: str, minimum: int) -> int:
        if not np.isfinite(value):
            raise ValueError(f"Non-finite WAVECAR {label}: {value!r}")
        integer = int(round(float(value)))
        if abs(float(value) - integer) > 1e-6 or integer < minimum:
            raise ValueError(f"Invalid WAVECAR {label}: {value!r}")
        return integer

    def _read_array(
        self,
        dtype: np.dtype | type,
        count: int,
        offset: int,
        label: str,
    ) -> np.ndarray:
        if self.f is None or self.f.closed:
            raise ValueError("WAVECAR is closed")
        self.f.seek(offset)
        values = np.fromfile(self.f, dtype=dtype, count=count)
        if values.size != count:
            raise EOFError(
                f"Short read for {label}: expected {count} values, got {values.size}"
            )
        return values

    def _validate_file_size(self, next_record: int) -> None:
        size = self.path.stat().st_size
        # Direct-access records are fixed length; ``next_record`` is the first
        # record after the final coefficient record of the first spin block.
        minimum = next_record * self.recl
        if size < minimum:
            raise EOFError(
                f"WAVECAR is truncated: {size} bytes, expected at least {minimum}"
            )

    @staticmethod
    def _nice_fft_size(value: int) -> int:
        candidate = value
        while True:
            remainder = candidate
            for factor in (2, 3, 5):
                while remainder % factor == 0:
                    remainder //= factor
            if remainder == 1:
                return candidate
            candidate += 1

    def _minimum_fft_grid(self) -> np.ndarray:
        gmax = np.asarray(
            [
                int(
                    np.sqrt(self.encut / HSQDTM)
                    / (2 * np.pi * np.linalg.norm(self.B[axis]))
                )
                + 1
                for axis in range(3)
            ],
            dtype=int,
        )
        return np.asarray(
            [self._nice_fft_size(int(2 * value + 2)) for value in gmax],
            dtype=int,
        )

    def _validate_k_index(self, ik: int) -> int:
        if isinstance(ik, bool) or not isinstance(ik, (int, np.integer)):
            raise TypeError(f"k-point index must be an integer, got {type(ik).__name__}")
        value = int(ik)
        if not 0 <= value < self.nk:
            raise IndexError(f"k-point index {value} outside 0..{self.nk - 1}")
        return value

    def _validate_band_index(self, ib: int) -> int:
        if isinstance(ib, bool) or not isinstance(ib, (int, np.integer)):
            raise TypeError(f"band index must be an integer, got {type(ib).__name__}")
        value = int(ib)
        if not 0 <= value < self.nb:
            raise IndexError(f"band index {value} outside 0..{self.nb - 1}")
        return value

    def gvecs(self, ik: int) -> np.ndarray:
        ik = self._validate_k_index(ik)
        if ik in self._gcache:
            return self._gcache[ik]
        nplane, kvector, _eigenvalues, _occupations = self.kinfo[ik]
        axes = []
        for raw_size in self.ng:
            size = int(raw_size)
            indices = np.arange(size, dtype=np.int64)
            axes.append(np.where(indices < size // 2 + 1, indices, indices - size))

        n1, n2, n3 = (len(axis) for axis in axes)
        candidates = np.empty((n1 * n2 * n3, 3), dtype=np.int64)
        # Preserve the WAVECAR order exactly: g3 outer, g2 middle, g1 inner.
        candidates[:, 0] = np.tile(axes[0], n2 * n3)
        candidates[:, 1] = np.tile(np.repeat(axes[1], n1), n3)
        candidates[:, 2] = np.repeat(axes[2], n1 * n2)

        cartesian = (candidates.astype(np.float64) + kvector) @ self.B
        kinetic = (
            HSQDTM
            * (2 * np.pi) ** 2
            * np.sum(cartesian * cartesian, axis=1)
        )
        result = candidates[kinetic < self.encut]
        if len(result) != nplane:
            raise ValueError(
                f"k-point {ik}: generated {len(result)} G-vectors, "
                f"WAVECAR declares {nplane}"
            )
        self._gcache[ik] = result
        return result

    def coeffs(self, ik: int, ib: int) -> np.ndarray:
        ik = self._validate_k_index(ik)
        ib = self._validate_band_index(ib)
        record = self.krec[ik] + 1 + ib
        nplane = self.kinfo[ik][0]
        return self._read_array(
            self.coefficient_dtype,
            nplane,
            record * self.recl,
            f"coefficients k={ik + 1} band={ib + 1}",
        )

    def psi_sq(self, ik: int, ib: int) -> np.ndarray:
        """Return normalized ``|psi(r)|^2`` on the minimum FFT grid."""

        vectors = self.gvecs(ik)
        coefficients = self.coeffs(ik, ib).astype(np.complex128)
        norm = float(np.sum(np.abs(coefficients) ** 2))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"Invalid coefficient norm for k={ik + 1}, band={ib + 1}")
        coefficients /= np.sqrt(norm)
        grid = np.zeros(tuple(int(value) for value in self.ng), dtype=np.complex128)
        grid[
            vectors[:, 0] % self.ng[0],
            vectors[:, 1] % self.ng[1],
            vectors[:, 2] % self.ng[2],
        ] = coefficients
        psi = np.fft.ifftn(grid) * np.prod(self.ng)
        density = np.abs(psi) ** 2
        density *= 1.0 / (density.sum() * self.vol / np.prod(self.ng))
        return density

    @property
    def closed(self) -> bool:
        return self.f is None or self.f.closed

    def close(self) -> None:
        if self.f is not None and not self.f.closed:
            self.f.close()

    def __enter__(self) -> "Wavecar":
        if self.closed:
            raise ValueError("Cannot enter a closed WAVECAR")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
