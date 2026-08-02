# Synthetic fluorescence transport data

Files in this directory are hand-constructed, non-empirical examples for unit
tests and API demonstrations. They are not measurements, calibrated material or
camera profiles, or evidence of physical-world performance. Do not use them for
paper results, physical claims, safety claims, or deployment decisions.

Publication experiments must supply a separate `calibration_type: "measured"`
document with traceable provenance and measurements on the same wavelength grid.
The loader intentionally provides no fallback calibration.
