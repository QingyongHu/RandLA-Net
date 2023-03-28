# Toronto-3D and OpenGF dataset code for RandLA-Net

Code for [Toronto-3D](https://github.com/WeikaiTan/Toronto-3D.git) has been uploaded. Try it for building your own network.

Will release code for OpenGF later

## Train and test RandLA-Net on Toronto-3D
1. Set up environment and compile the operations - exactly the same as the RandLA-Net environment
1. Create a folder called `data` and move the `.ply` files into `data/Tronto_3D/original_ply/`
1. Change parameters according to your preference in `data_prepare_toronto3d.py` and run to preprocess point clouds
1. Change parameters according to your preference in `helper_tool.ply` to build the network
1. Train the network by running `python main_Toronto3D.py --mode train`
1. Test and evaluate on `L002` by running `python main_Toronto3D.py --mode test --test_eval True`
1. Modify the code to find a good parameter set or test on your own data

## Sample results of Toronto-3D
The highest results reported are from [Hu et al. (2021)](https://doi.org/10.1109/TPAMI.2021.3083288). Here are some results I got on my code with the default parameters. The largest factor in mIoU is the accuracy of *Road Markings*, which is impossible to be classified with XYZ only.

| Features | OA | mIoU | Road | Road mrk. | Natural | Bldg | Util. line | Pole | Car | Fence |
|----------|----|------|------|-----------|---------|------|------------|------|-----|-------|
| [RandLA-Net](https://doi.org/10.1109/TPAMI.2021.3083288) | 92.95 | 77.71 | 94.61 | 42.62 | 96.89 | 93.01 | 86.51 | 78.07 | 92.85 | 37.12 |
| [RandLA-Net](https://doi.org/10.1109/TPAMI.2021.3083288) (RGB)| 94.37 | 81.77 | 96.69 | 64.21 | 96.92 | 94.24 | 88.06 | 77.84 | 93.37 | 42.86 |
|XYZRGBI | 96.57 | 81.00 | 95.61 | 58.04 | 97.22 | 93.45 | 87.58 | 82.64 | 91.06 | 42.39 |
|XYZRGB  | 96.71 | 80.89 | 95.88 | 60.75 | 97.02 | 94.04 | 86.71 | 83.30 | 87.66 | 41.80 |
|XYZI    | 95.65 | 80.03 | 94.59 | 50.14 | 95.90 | 92.76 | 87.70 | 77.77 | 91.10 | 50.30 |
|XYZ     | 94.94 | 74.13 | 93.53 | 12.52 | 96.67 | 92.34 | 86.25 | 80.10 | 88.04 | 43.57 |

