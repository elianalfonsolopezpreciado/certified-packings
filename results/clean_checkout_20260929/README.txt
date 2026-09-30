Clean-checkout test, 2026-09-29 (UTC-6): `git clone` of the repository (commit 275c174, first v1.0 candidate) into a temporary folder, then
  make test        -> 51 passed (test_exit=0)
  make verify      -> 84 certificates, 168/168 checks valid (both backends, fresh processes) (verify_exit=0)
  make reproduce   -> headline table + figures regenerated from certificates/ only (reproduce_exit=0)
  python validator/validate_pck.py submission/csqv{120,250}.pck --require-contacts -> both valid, 360/360 and 750/750 contacts
  python scripts/verify_shuffled.py --rounds 3 -> 3 rounds, different process order and PYTHONHASHSEED (1,2,3), 84 certificates each: "ALL CONSISTENT"
Later commits only add documentation, release scripts, the email draft and clarifications (no change to code paths used by these checks).
