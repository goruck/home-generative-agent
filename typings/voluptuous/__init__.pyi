# Type-checking only: tells pyright that ``voluptuous`` is probatio.
#
# From 2026.9 Home Assistant aliases probatio under the ``voluptuous`` name at
# startup (``install_as_voluptuous()`` in ``homeassistant/__init__.py``), and
# from 2026.10 core annotates its APIs with ``probatio`` types. Pyright cannot
# see that runtime alias and resolves ``import voluptuous`` to the real
# voluptuous package other dependencies still install, so every schema we hand
# core reads as the wrong type. Every name we use is the same object in both at
# runtime (checked against 2026.10). The code keeps ``import voluptuous`` because
# HA 2026.8, the oldest supported core, does not ship probatio.
from probatio import *  # noqa: F403 -- the whole surface, by design
