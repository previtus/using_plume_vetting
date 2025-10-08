# How to use?

1.) Install required libraries (the usual suspects, plus "pip install georeader").

2.) Run **main_plume_scoring_demo.py**

Optionally change the name of the loaded EMIT tile (ps: example vector predictions available only for the default file).
Also optionally turn on/off visualisation with "debug_viz=True"
The run_plume_vetting_on_scene function returns dictionary with scores (D_Norm and Estimated_concentration_len) assigned to each separate vector given to it (small vectors can be ignored with "min_polygon_size = 0")

3.) ... probably rewrite with your own data loaders and later use as a function in a prediction loop ...

NOTE: Current implementation just scores all provided polygons, it doesn't run analysis of any potential background samples (as was done in the source paper).
