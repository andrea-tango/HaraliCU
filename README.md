# HaraliCU

HaraliCU is a GPU-powered approach designed to efficiently computate the Gray-Level Co-occurrence Matrix (GLCM) in order to extract an exhaustive set of the Haralick features.

It has been conceived to overcome the limitations of the existing feature extraction and radiomics tools that cannot effectively manage the full-dynamics of gray-scale levels (i.e., 16 bit). 

This novel approach is a promising solution that might enable multi-scale radiomic analyses by properly combining several values of distance offsets, orientations, and window sizes.

  1. [References](#ref) 
  2. [Required libraries](#lib) 
  3. [Input parameters](#inp)
  4. [Datasets](#data)
  5. [License](#lic)
  6. [Contacts](#cont)
  
## <a name="ref"></a>References ##

A detailed description of HaraliCU, as well as a complete experimental comparison against a CPU version by using the dataset described below ([Data](#data)), can be found in:

- Rundo L., Tangherloni A., Galimberti S., Cazzaniga P., Woitek R., Sala E., Nobile M.S., Mauri G.: "HaraliCU: GPU-Powered Haralick FeatureExtraction on Medical Images Exploitingthe Full Dynamics of Gray-scale Levels", International Conference on Parallel Architectures and Compilation Techniques, submitted.



## <a name="lib"></a>Required libraries ##


## <a name="inp"></a>Input parameters ##

In oder to run HaraliCU, the following parameters must be provided:

- `-i` to specify the input image to process (in TIFF/TIF);
  
Optional parameters could be provided:

## <a name="datasets"></a>Data ##

## <a name="lic"></a>License ##

MedGA is licensed under the terms of the GNU GPL v3.0

## <a name="cont"></a>Contacts ##

For questions or support, please contact <andrea.tangherloni@disco.unimib.it> (<at860@cam.ac.uk>)
and/or <leonardo.rundo@disco.unimib.it> (<lr495@cam.ac.uk>).
