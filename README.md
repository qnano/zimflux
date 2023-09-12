# ZIMFLUX

Welcome to the ZIMFLUX repository! ZIMFLUX is a fluorescence microscopy project that combines modulated excitation patterns with Point Spread Function (PSF) engineering. If you're specifically interested in the vectorial PSF model, you might want to explore our related repository, [VectorialPSF](https://github.com/pvanvelde/VectorialPSF). 

For the development of ZIMFLUX, we leverage several powerful tools and functionalities from [PhotonPy](https://github.com/qnano/photonpy). These include fast spot detection and drift correction, enhancing the overall capabilities and precision of our project.

## Table of Contents

- [Requirements](#requirements)
- [Installation and Usage](#installation-and-usage)
- [Possible Fixes](#possible-fixes)
- [License](#license)

## Requirements

Before you begin, ensure you have the following prerequisites:

- **Operating System:** Windows (Tested on Windows 10)
- **CUDA Compatible GPU:** It's essential to have a CUDA compatible GPU (tested on NVIDIA GeForce GTX 980). You can find compatibility information [here](https://docs.nvidia.com/deploy/cuda-compatibility/).


## Installation and Usage

Follow these steps to install and set up the project:

1. **Install Python:**
   - We recommend using Anaconda to manage Python environments. You can download Anaconda [here](https://www.anaconda.com/distribution/).

2. **Create a Virtual Environment:**
   - Create a virtual environment using Anaconda with the following commands:
     ```bash
     conda create -n myenv anaconda python=3.8
     conda activate myenv
     ```

3. **Install Visual Studio 2019 Community Edition:**
   - Download and install Visual Studio 2019 Community Edition from [here](https://visualstudio.microsoft.com/vs/older-downloads/).

4. **Extract External Libraries:**
   - Extract the external libraries from `cpp/external.zip` so that the `cpp` directory contains a folder called "external."

5. **Install CUDA Toolkit 11.3:**
   - Install [CUDA Toolkit 11.3](https://developer.nvidia.com/cuda-downloads).
   - Check the installation guide [here](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-installation-guide-microsoft-windows/index.html).

6. **Build SMLMLib:**
   - In Visual Studio, open the `smlm.sln` solution.
   - Set the build mode to "**Release**" and platform to "**x64**."
   - Build the `SMLMLib` project.

7. **Install Python Dependencies:**
   - Install the required Python packages by running:
     ```bash
     pip install -r requirements.txt
     ```

8. **Install PyTorch:**
   - Install PyTorch with the following command:
     ```bash
     pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
     ```

9. **Download Datasets:**
   - Download datasets from Surfdrive [here](https://surfdrive.surf.nl/files/index.php/s/Ygu3FBLX4jPbMly).

10. **Run the code on a 3D Nanoruler Dataset:**
    - Change the 'absolute path to data' and 'absolute path to gain and offset data' to match the location where you stored the downloaded files from Surfdrive.
    - You should now be able to run the script `ZIMFLUX/runfile.py`.

11. **Results**
    - Most figures are saved in the folder where the data is located. The .hdf5 files are compatible with [Picasso](https://github.com/jungmannlab/picasso).

## Possible Fixes

If you encounter issues or the above steps do not yield the desired results, consider these fixes:

1. **Check CUDA Installation:**
   - Run the following command in the terminal to check if CUDA is installed correctly:
     ```
     nvidia-smi
     ```

2. **Verify CUDA Availability with PyTorch:**
   - Open a Python interpreter and run the following commands to check if CUDA is detected by PyTorch:
     ```python
     import torch
     torch.cuda.is_available()
     ```

3. **Ensure Correct CUDA Version:**
   - Verify that your CUDA version is compatible with your PyTorch version. Ensure you have CUDA 11.3 installed for compatibility.

4. **Change CUDA Settings:**
   - Search for "NSight Monitor" and run it as an administrator.
   - It may not open a new window but appear as a background process in the system tray (bottom right part of your screen).
   - Right-click the NSight Monitor icon and go to 'Options.'
   - Change 'WDDM TDR Delay' to '180' and 'WDDM TDR Enabled' to 'False.'
   - You may need to reboot your system.

5. **Update GPU Driver:**
   - Search for 'Device Manager.'
   - Under 'Display adapters,' right-click your video card (usually NVIDIA XXXX).
   - Click 'Properties' -> 'Driver' -> 'Update Driver...'
   - Follow the prompts to update the GPU driver.
   - You may need to reboot your system.

## License

This project is licensed under the MIT license - see the [LICENSE](LICENSE.txt) file for details.
