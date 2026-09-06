"""Ingest layer.

Split deliberately into two halves:

  Pure transforms -- `beam`, `profile`, and the parsing helpers in `nexrad`
  and `glm` -- operate on arrays and text, and are covered by
  tests_ingest.py.

  Fetch and decode -- the functions marked UNVERIFIED below -- need network
  access plus Py-ART and netCDF4, so they are written against the documented
  APIs but have not been run against a real file. Shake them out on first
  use; do not assume they work because the tests pass.
"""
