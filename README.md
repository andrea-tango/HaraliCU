# HaraliCU

HaraliCU is a GPU-powered approach designed to efficiently compute the Gray-Level Co-occurrence Matrix (GLCM) in order to extract an exhaustive set of the Haralick features.

It has been conceived to overcome the limitations of the existing feature extraction and radiomics tools that cannot effectively manage the full-dynamics of grey-scale levels (i.e., 16 bit).

This novel approach is a promising solution that might effectively enable multi-scale radiomic analyses by properly combining several values of distance offsets, orientations, and window sizes.

  1. [References](#ref) 
  2. [Required libraries](#lib) 
  3. [Input parameters](#inp)
  4. [Datasets](#data)
  5. [License](#lic)
  6. [Contacts](#cont)
  
## <a name="ref"></a>References ##

A detailed description of HaraliCU, as well as a complete experimental comparison against the corresponding CPU version by using the dataset described below ([Data](#data)), can be found in:

- Rundo L., Tangherloni A., Galimberti S., Cazzaniga P., Woitek R., Sala E., Nobile M.S., Mauri G.: "HaraliCU: GPU-Powered Haralick Feature Extraction on Medical Images Exploitingthe Full Dynamics of Gray-scale Levels", In: Proc. 15th International Conference on Parallel Computing Technologies (PaCT) 2019. (Accepted Manuscript)

## <a name="lib"></a>Required libraries ##

The CPU version of HaraliCU has been developed in `C++`, whilst the GPU version in `CUDA C++`.
Both versions have been developed and tested on Ubuntu Linux but should work also on MacOS X and Windows.

HaraliCU depends on:

- `CUDA` toolkit;
- `OpenCV`.

In order to compile both versions of HaraliCU, we provide two bash scripts (_compileCPU.sh_ and _compileGPU.sh_).

The resulting executable files (named HaraliCU and HaraliCPU) are standalone executable programs.

## <a name="inp"></a>Input parameters ##

In oder to run HaraliCU, the following parameters must be provided:

- `-i` to specify the input folder containing the images to process (in BMP, DIB, JPEG, JPG, JPE, JP2, PNG, WEBP, PBM, PGM, PPM, SR, RAS, TIFF, TIF).

  
Optional parameters could be provided:
- `-o` to specify the output folder (default: output);
- `-w` to specify the window size (default: 5);
- `-p` to specify the padding strategy (default: 1 = zero padding. 0 = no padding; 2 = symmetric);
- `-d` to specify the distance between the reference pixel and the neighbor pixel (default: 1); 
- `-t` to specify the orientation (default: 1 = 0°. 2 = 45°; 3 = 90°; 4 = 135°);
- `-q` to specify the maximum gray level for quantization (default: the maximum value in the image);
- `-g` to make the GLCM pairs symmetric;
- `-s` to save the feature maps;
- `-v` to enable the verbose modality.


By running python HaraliCU without specifying any parameter (or using `-?` or `-h`), all the above parameters will be listed.

## <a name="datasets"></a>Data ##

## <a name="lic"></a>License ##

HaraliCU is licensed under the terms of the GNU GPL v3.0

## <a name="cont"></a>Contacts ##

For questions or support, please contact <andrea.tangherloni@disco.unimib.it> (<at860@cam.ac.uk>)
and/or <leonardo.rundo@disco.unimib.it> (<lr495@cam.ac.uk>).
