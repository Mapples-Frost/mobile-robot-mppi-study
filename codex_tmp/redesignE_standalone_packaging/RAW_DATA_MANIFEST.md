# Redesign E raw-data integrity manifest

The standalone raw tree contains 1,408 files totaling 1,934,788,074 bytes.
Every file was re-read after packaging and verified against
`SOURCE_RAW_DATA.sha256`.

- verified files: 1,408/1,408
- missing files: 0
- checksum mismatches: 0
- unexpected files: 0
- verification: PASSED

Each arm contains 50 matched case directories. Each case directory contains
seven evidence files: case identity, resolved configuration, configuration hash,
metrics, preflight, provenance and compressed trajectory.
