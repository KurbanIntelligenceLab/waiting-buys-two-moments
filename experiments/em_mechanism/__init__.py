"""Why does early-stopped EM beat converged NPMLE?  Trajectory study.

See experiments/em_mechanism/README.md for the protocol and reproduction
commands.  Nothing here changes `wtm/`; the existing estimators are reused
and, where this package re-implements the EM update in batch form, the
re-implementation is checked against `wtm.npmle.fit_npmle` (validate.py).
"""
