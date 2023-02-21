# Toronto-3D and OpenGF dataset code for RandLA-Net

Code for [Toronto-3D](https://github.com/WeikaiTan/Toronto-3D.git) has been uploaded. Try it for building your own network.

Will work on code for OpenGF

## Train and test RandLA-Net on Toronto-3D
1. Set up environment and compile the operations - exactly the same as the RandLA-Net environment
1. Create a folder called `data` and move the `.ply` files into `data/Tronto_3D/original_ply/`
1. Change parameters according to your preference in `data_prepare_toronto3d.py` and run to preprocess point clouds
1. Change parameters according to your preference in `helper_tool.ply` to build the network
1. Train the network by running `python main_Toronto3D.py --mode train`
1. Test and evaluate on `L002` by running `python main_Toronto3D.py --mode test --test_eval True`
1. Modify the code to find a good parameter set or test on your own data
