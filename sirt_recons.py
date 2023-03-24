
from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.algorithms import SIRT

data_clean = CentreOfRotationCorrector.image_sharpness('centre',\
                                                            backend='astra',\
                                                        search_range=250, \
                                                            tolerance=0.005)(data_clean)
data_clean2 = Slicer(roi={'angle': (0,-1,2)})(data_clean)
data_clean4 = Slicer(roi={'angle': (0,-1,4)})(data_clean)
data_clean8 = Slicer(roi={'angle': (0,-1,8)})(data_clean)

ig2D = data_clean.geometry.get_ImageGeometry()
ag2D = data_clean.geometry

ig2D2 = data_clean2.geometry.get_ImageGeometry()
ag2D2 = data_clean2.geometry

ig2D4 = data_clean4.geometry.get_ImageGeometry()
ag2D4 = data_clean4.geometry

ig2D8 = data_clean8.geometry.get_ImageGeometry()
ag2D8 = data_clean8.geometry

A = ProjectionOperator(ig2D,ag2D, device="gpu")
A2 = ProjectionOperator(ig2D2,ag2D2, device="gpu")
A4 = ProjectionOperator(ig2D4,ag2D4, device="gpu")
A8 = ProjectionOperator(ig2D8,ag2D8, device="gpu")


x0 = ig2D.allocate(0.0)
x02 = ig2D2.allocate(0.0)
x04 = ig2D4.allocate(0.0)
x08 = ig2D8.allocate(0.0)
sirt_nz = SIRT(initial=x0, operator=A, data=data_clean, max_iteration=3000, lower=0.0, update_objective_interval=50)
sirt_nz2 = SIRT(initial=x02, operator=A2, data=data_clean2, max_iteration=3000, lower=0.0, update_objective_interval=50)
sirt_nz4 = SIRT(initial=x04, operator=A4, data=data_clean4, max_iteration=3000, lower=0.0, update_objective_interval=50)
sirt_nz8 = SIRT(initial=x08, operator=A8, data=data_clean8, max_iteration=3000, lower=0.0, update_objective_interval=50)














