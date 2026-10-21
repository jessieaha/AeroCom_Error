"""Backward-compatible import path.

The implementation lives in ``regional_aggregates``. New code should call
``regional_aggregates.build_regional``.
"""

from regional_aggregates import *  # noqa: F401,F403
from regional_aggregates import (  # noqa: F401
    AEROCOM_QUALITY,
    AEROCOM_SEASONAL,
    QUALITY_FILTER_PARAMS,
    build_regional,
    main,
)

if __name__ == "__main__":
    import sys
    sys.exit(main())
