from .kde import *
from .maf import *
from .vae import *
from .cdf import *
from .ldm import *
from .ttsgan import *

models = {
    'KDE': KDE,
    'MAF': MAF,
    'VAE': VAE,
    'CDF': CDF,
    'LDM': LDM,
    'TTSGAN': TTSGAN,
}
