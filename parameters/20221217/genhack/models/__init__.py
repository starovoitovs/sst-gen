from .generative.kde import *
from .generative.maf import *
from .generative.vae import *
from .generative.cdf import *
from .generative.cnf import *
from .generative.ldm import *
from .generative.ttsgan import *
from .generative.gpr import *
from .ts import *
from .weights import *

models = {
    'KDE': KDE,
    'MAF': MAF,
    'VAE': VAE,
    'CDF': CDF,
    'CNF': CNF,
    'LDM': LDM,
    'TTSGAN': TTSGAN,
    'GPR': GPR,
}

ts_models = {
    'TrendModel': TrendModel,
}

weights_models = {
    'LearnableWeights': LearnableWeights,
    'PowerLawWeights': PowerLawWeights,
}
