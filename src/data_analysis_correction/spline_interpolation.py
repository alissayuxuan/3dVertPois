import numpy as np
from scipy.interpolate import CubicSpline
from TPTBox import Location

# put spline interpolation through the corpus points and measure shift against shift in corpus COMS of vertebra neighbors
# maybe start without spline interpolation but just check POI coordinate shift against COM shift

corpus_COM_L = Location.Vertebra_Corpus.value
# ALL


# shift corpus ALL POI anoterior then s proj again
