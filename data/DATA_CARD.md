# Dataset and asset card

Status: **incomplete — do not publish or claim artifact reproducibility until every
required field below is filled and reviewed.** This file intentionally does not
infer provenance, consent, licenses, or measurement conditions from filenames.

## Asset inventory

For each sign image, active-state image, pole image, background collection,
physical photo set, and video set, record:

| Asset or glob | Source/collector | Acquisition date | License/consent | Allowed redistribution | SHA-256 manifest |
|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO |

## Collection protocol

- Location class and whether public/private property was involved: TODO
- Camera make/model, lens, resolution, codec, and firmware: TODO
- Exposure mode and retained metadata: TODO
- Sign/replica dimensions, substrate, retroreflective material, and mounting: TODO
- Paint manufacturer/product, batch, application method, thickness/volume, and curing: TODO
- Excitation wavelength, lamp model/power, beam spread, and irradiance at sign: TODO
- Measured lux, weather, distance, yaw/pitch, vehicle speed, and trial identifiers: TODO
- Number of independent signs, material batches, cameras, days, and repeated runs: TODO
- Train/validation/certification split policy and leakage checks: TODO

## Privacy and ethics

Document faces, license plates, bystanders, precise locations, removal/redaction,
institutional review, permission to photograph/modify signs, and safe testing
controls. Experiments must use owned or expressly authorized signs and must not
alter deployed public traffic-control infrastructure.

## Known limitations

TODO. At minimum distinguish static frames from continuous approaches, controlled
replicas from public-road signs, estimated from measured metadata, and simulation
colors from spectrally calibrated material/camera measurements.

## Machine-readable manifest

Before release, add a versioned CSV or JSON manifest with one row per independent
capture (not merely one row per extracted frame), stable IDs, split membership,
all measured covariates, checksums, and explicit missing-value markers. Statistical
analysis should cluster frames by physical run/video.
