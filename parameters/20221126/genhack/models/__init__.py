from .kde import *
from .maf import *
from .vae import *
from .cdf import *
from .ldm import *

models = {
    'KDE': KDE,
    'MAF': MAF,
    'VAE': VAE,
    'CDF': CDF,
    'LDM': LDM,
}
