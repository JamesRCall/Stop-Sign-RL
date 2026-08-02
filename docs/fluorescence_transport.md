# Fluorescence-to-camera transport

`utils/fluorescence_transport.py` is a pure, deterministic radiometric boundary
between calibrated optical measurements and the learning system. It does not load
a detector, alter an environment, or supply guessed physical constants.

## Model

All spectra use one strictly increasing wavelength grid in nanometres. For
ambient spectral irradiance `E_a`, trigger spectral irradiance `E_u`, substrate
reflectance `rho`, and coating transmittance `tau`, the Lambertian components are

```text
L_day(lambda)     = E_a(lambda) rho(lambda) tau(lambda) / pi
L_trigger(lambda) = [E_a(lambda) + E_u(lambda)] rho(lambda) tau(lambda) / pi
```

The fluorescent term has two mutually exclusive calibrated forms:

- `excitation_emission`: integrate trigger irradiance against measured excitation
  efficiency, apply radiant efficiency, and distribute the energy over a
  normalized measured emission spectrum;
- `measured_emission_response`: scale a directly measured spectral-radiance
  response by integrated trigger irradiance.

The triggered state adds this fluorescent radiance to `L_trigger`. Camera channel
responses are trapezoidal wavelength integrals of spectral radiance times the
camera sensitivity curves and exposure scale. A calibrated 3-by-3 ISP matrix and
offset then produce linear RGB. The API returns unclipped RGB, clipped RGB, and
per-channel clipping flags; it performs no gamma encoding.

## Calibration contract

The versioned contract is
[`schemas/fluorescence_transport_v1.schema.json`](../schemas/fluorescence_transport_v1.schema.json).
The loader additionally enforces relationships JSON Schema cannot express
conveniently, including equal spectral lengths, a strictly increasing shared
grid, ordered bounds, positive spectral integrals, and uncertainty bounds that
cannot move reflectance, transmittance, excitation efficiency, or camera
sensitivity outside `[0, 1]`.

Every calibration must explicitly provide:

- units, wavelength samples, ambient and trigger irradiance;
- substrate reflectance and material transmittance;
- one fluorescence formulation;
- camera RGB sensitivity, exposure scale, ISP matrix, and RGB offset;
- bounded, independent uniform uncertainty for material batch, camera, and
  irradiance factors; and
- provenance declaring whether the values are measured or synthetic.

Unknown keys, duplicate JSON keys, non-finite values, unsupported versions, and
missing uncertainty fields are rejected. There is deliberately no built-in
physical calibration and no interpolation or inferred unit conversion.

## Reproducibility and provenance

`compute_fluorescence_transport(calibration, seed=...)` requires an unsigned
64-bit seed. For a fixed calibration and seed, uncertainty draws and RGB outputs
are deterministic. Results carry two SHA-256 values:

- `calibration_sha256` hashes canonical JSON content, independent of formatting;
- `calibration_source_sha256` hashes the exact source bytes.

Store both hashes and the seed with experiment artifacts. The draw itself is also
returned so a result can be audited without reconstructing pseudo-random state.

## Synthetic example boundary

[`data/synthetic/fluorescence_transport_v1.synthetic.json`](../data/synthetic/fluorescence_transport_v1.synthetic.json)
is hand-constructed only to test equations, validation, and deterministic I/O. It
is prominently marked `synthetic_non_empirical`; it is not evidence for physical
effectiveness or a substitute for material, illumination, and camera calibration.
Publication results should use traceable `measured` calibration documents and
archive their hashes with the reported experiment.
