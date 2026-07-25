# Single-dynamic-obstacle paper-v4 formal analysis

This directory preserves the complete ten-file formal analysis bundle copied
from the Windows ECS host after all 360 blocks and 1,860 episode jobs completed.

- Transport archive SHA-256:
  `FF1C945942278B195D582435726B3F7557838E873F07357D3588099263B6DFED`
- Analysis bundle SHA-256 (canonical file-hash map):
  `196260f274983f41e4e5f9de959a210fd9ec41193248f0ab757b97ab5255e1bc`
- Sealed registry SHA-256:
  `91cc59c6025296bcb4aa5f8839bd5bc129159c109730f52c5c9ac13d03b47b56`
- Original analysis script SHA-256:
  `7424485CD2730754ECBCD100E666E105E6CF571C435419F3385A1DA2CB1BAD46`
- Pairing-corrected analysis script SHA-256:
  `76E55FC5CD1A4CA6EE9BBAD3ADC44FA012DD5FC0CA39D4BDCCFFE1BDC90B1962`

The first two analysis attempts opened outcomes only after the complete-matrix
audit, then stopped when the ablation analysis incorrectly attempted to pair
the Full arm's 360 seeds with the frozen 140-seed ablation subset.  The bounded
correction restricted both sides of the ablation contrasts to registry-derived
`ablation_block` rows.  It did not modify formal episodes, seeds, endpoints,
bootstrap settings, multiplicity families, or primary contrasts.  The original
execution mirror and both failed analysis directories remain preserved on the
ECS host pending final archival.

`analysis/analysis_bundle_manifest.json` authenticates the other nine analysis
files.  `paper_v4_analysis_transport.zip` preserves the exact archive received
from the ECS host.
