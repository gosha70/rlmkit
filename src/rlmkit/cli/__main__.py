"""Allow ``python -m rlmkit.cli`` as an alternative to the ``rlmkit``
console script.

The two entry points are equivalent. The console script is the
documented public interface; the ``python -m`` form is useful when the
script's bin directory is not on ``PATH`` (some virtualenv-in-a-virtualenv
setups, some CI runners).
"""

from rlmkit.cli.main import main

raise SystemExit(main())
