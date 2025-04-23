from .amstertime import AmsterTime
from .eynsham import Eynsham
from .mapillary import MSLS
from .nordland import NordlandDataset
from .pittsburgh import Pitts30k, Pitts250k
from .svox import SVOX, SVOXNight, SVOXOvercast, SVOXRain, SVOXSnow, SVOXSun
from .tokyo import Tokyo247

ALL_DATASETS = {
    "tokyo247": Tokyo247,
    "amstertime": AmsterTime,
    "eynsham": Eynsham,
    "msls": MSLS,
    "nordland": NordlandDataset,
    "pitts30k": Pitts30k,
    "pitts250k": Pitts250k,
    "svox_night": SVOXNight,
    "svox_rain": SVOXRain,
    "svox_sun": SVOXSun,
    "svox_snow": SVOXSnow,
    "svox_overcast": SVOXOvercast,
}
