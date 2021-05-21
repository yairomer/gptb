import logging

from . import notebook  # noqa F401
from . import math  # noqa F401
from . import networks  # noqa F401
from . import datasets  # noqa F401
from . import visualizations  # noqa F401
from . import webaggserver  # noqa F401
from . import config  # noqa F401
from . import hmc  # noqa F401
from .inception import ISCalculator  # noqa F401
from .auxil import Profiler # noqa F401
from .auxil import set_random_state # noqa F401
from .auxil import CountDowner # noqa F401
from .auxil import Resevior # noqa F401
from .auxil import download_url_to_file # noqa F401

logger = logging.getLogger('gptb_basic')
logger.setLevel(logging.DEBUG)
logger.propagate = False

handler = logging.StreamHandler()
# handler.setFormatter(logging.Formatter('{message}', style='{'))
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
