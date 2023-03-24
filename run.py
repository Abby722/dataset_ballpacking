import sys
sys.path.append("../scripts/python")
import amglib.imageutils as utils
import amglib.readers as io
from tqdm.notebook import tqdm
sys.path.append("../sarepy")
sys.path.append("../algotom")
sys.path.insert(0, "path-to-sarepy-pck")
import numpy as np
import matplotlib.pyplot as plt
import skimage as im
import scipy.io
import os
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse

import importlib

dc=io.read_images(datapath+'dc_{:05}.fits',first=1,last=5,averageStack=True,average='mean');
ob=io.read_images(datapath+'ob_{:05}.fits',first=1,last=5,averageStack=True,average='mean');
ImgRaw = io.read_images(datapath+'Nail-soil-artificial-xct-Cu1.5_{:05}.fits',first=1,last = 1126,stride=1); 


# Import all CIL components needed
from cil.framework import ImageData, ImageGeometry
from cil.framework import AcquisitionGeometry, AcquisitionData

# CIL Processors
from cil.processors import CentreOfRotationCorrector, Slicer, TransmissionAbsorptionConverter, Normaliser, Padder

# CIL display tools
from cil.utilities.display import show2D, show_geometry

# From CIL ASTRA plugin
from cil.plugins.astra import ProjectionOperator, FBP

from cil.processors import RingRemover
from cil.utilities.jupyter import islicer, link_islicer
from cil.processors import Masker, MaskGenerator

import logging
logging.basicConfig(level=logging.WARNING)
cil_log_level = logging.getLogger('cil.processors')
cil_log_level.setLevel(logging.INFO)

# preview of raw data
fig, ax = plt.subplots(2,2,figsize=(15,10))
ax=ax.ravel()
ax[0].imshow(dc)
ax[0].set_title('DC')
a1=ax[1].imshow(ob)
fig.colorbar(a1,ax=ax[1])
ax[1].set_title('OB')
a2=ax[2].imshow(ImgRaw[300,:,:]) #,vmin=0.0,vmax=10000)
fig.colorbar(a2,ax=ax[2])
ax[2].set_title('Projection');
a3=ax[3].imshow(ImgRaw[300,0:20,10:50]) #,vmin=0.0,vmax=10000)
fig.colorbar(a3,ax=ax[3])
ax[3].set_title('does ROI');

# Normalisation
ImgNorm = utils.normalizeImage(ImgRaw, ob, dc, neglog = True, doseROI = [0,0,10,10])
ImgNorm= ImgNorm.astype('float32')

agRaw = AcquisitionGeometry.create_Parallel3D()  \
         .set_panel(num_pixels=(2047, 2047))        \
         .set_angles(angles=np.linspace(0, 360, num=1126, endpoint=False))
agRaw.set_labels(['angle','vertical','horizontal'])
print(agRaw.dimension_labels)

data3D = AcquisitionData(ImgNorm, geometry=agRaw, deep_copy=False)
data3D_roi = Slicer(roi={'vertical': (600, 1500), 'horizontal':(90,1890)})(data3D)
islicer(data3D_roi, minmax=(0,1.2), cmap=cmap)
del data3D

# 2D slice
s400 = data3D_roi.get_slice(vertical=400)

## Preprocessing
import algotom.prep.removal as srm
sino0 = utils.spotclean(s400.as_array(),size=5,threshold=0.90)
sino0 = sino0.astype('float32')
show2D([sino0, s400.as_array()-sino0],
       ['spot clean', 'difference'],
      )

sino1 = srm.remove_dead_stripe(sinogram=sino0, 
                               snr=3, size=41, residual=True, 
                               smooth_strength=30 # Window size of the uniform filter used to detect stripes.
                              )
sino1 = sino1.astype('float32')
show2D([sino1, sino0-sino1],
       ['remove dead stripe', 'difference'],
      )

sino2 = srm.remove_large_stripe(sino1, snr=3,
                                size = 35,
                                drop_ratio=0.1,norm=False,
                                #options={"method": "gaussian_filter", "para1":(1,42)}
                                            )
sino2 = sino2.astype('float32')
show2D([sino2, sino1-sino2],
       ['remove large stripe', 'difference'],
      )

sino3 = srm.remove_stripe_based_sorting(sino2, size=13,
                                       dim=2, 
                                        #options={"method": "gaussian_filter", "para1":(1,11)}
                                       )
sino3 = sino3.astype('float32')

show2D([sino3, sino2-sino3],
       ['sorting remove', 'difference'],
      )

sino4 = srm.remove_stripe_based_wavelet_fft(sino3, level=3, size=1,
                                           wavelet_name="db9",
                                           window_name="gaussian",
                                           sort=False
                                           )
sino4 = sino4.astype('float32')

show2D([sino4, sino3-sino4],
       ['wavelet remove', 'difference'],
      )
## Fill in the dataContainer
s400_clean = s400.clone()
s400_clean.fill(sino3)

### center of rotation
s400_clean = CentreOfRotationCorrector.xcorrelation(slice_index='centre', projection_index=10, ang_tol=0.5)(s400_clean)
print('data_centred rotation axis position: {}'.format(s400_clean.geometry.config.system.rotation_axis.position))



from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.algorithms import SPDHG
from cil.optimisation.algorithms import SIRT
from cil.optimisation.algorithms import CGLS
from cil.optimisation.algorithms import GD
from cil.optimisation.functions import LeastSquares
from cil.optimisation.algorithms import GD, FISTA, PDHG
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.optimisation.functions import IndicatorBox, MixedL21Norm, L2NormSquared, \
                                       BlockFunction, L1Norm, LeastSquares, \
                                       OperatorCompositionFunction, TotalVariation, \
                                       ZeroFunction
from cil.optimisation.functions import L2NormSquared, BlockFunction, MixedL21Norm, IndicatorBox
from cil.optimisation.operators import GradientOperator, BlockOperator
from cil.optimisation.algorithms import PDHG


data2 = Slicer(roi={'angle': (0,-1,5)})(s800_clean)
ig2D = data2.geometry.get_ImageGeometry()
ag2D = data2.geometry

A = ProjectionOperator(ig2D,ag2D, device="gpu")
x0 = ig2D.allocate(0.0)


# FBP reconstruction
fbp_recon = FBP(ig2D, 
                     ag2D, device='gpu')(data2)
# SIRT reconstruction
sirt_nz = SIRT(initial=x0, operator=A, data=data2, max_iteration=3000, lower=0.0, update_objective_interval=50)
sirt_nz.run(2000, verbose=1)

# CGLS reconstruction
cgls = CGLS(initial=x0, 
            operator=A, 
            data=data2,
            max_iteration = 150,
            update_objective_interval = 10 )

cgls.run()



# set up SPDHG
data = data2

# Define number of subsets
n_subsets = 10

# Initialize the lists containing the F_i's and A_i's
f_subsets = []
A_subsets = []

# Define F_i's and A_i's
for i in range(n_subsets):
    # Total number of angles
    n_angles = len(ag2D.angles)
    # Divide the data into subsets
    data_subset = Slicer(roi = {'angle' : (i,n_angles,n_subsets)})(data)
    # Define F_i and put into list
    fi = 0.5*L2NormSquared(b = data_subset)
    f_subsets.append(fi)
    # Define A_i and put into list 
    ageom_subset = data_subset.geometry
    Ai = ProjectionOperator(ig2D, ageom_subset, device='gpu')
    A_subsets.append(Ai)

# Define F and K
F = BlockFunction(*f_subsets)
K = BlockOperator(*A_subsets)

# Define G (by default the positivity constraint is on)
alpha = 0.01
G = alpha * FGP_TV()

spdhg = SPDHG(f = F, g = G, operator = K,  max_iteration = 100,
            update_objective_interval = 5)
spdhg.run()

### TV denoise
alpha = 0.012
TV = alpha * TotalVariation(max_iteration=7)
fbpTV = TV.proximal(fbp_recon, tau=0.1)
show2D([fbp_recon_s800, fbpTV,fbp_recon], 
       title=['fbp full','Total variation', 'fbp reduced'], 
       origin="upper", num_cols=3,
      cmap = cmap, fix_range=(0,.004))
