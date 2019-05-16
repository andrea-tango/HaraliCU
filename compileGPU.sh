nvcc GPU/src/AggregatedGrayPair.cu GPU/src/Direction.cu GPU/src/Features.cu GPU/src/GrayPair.cu GPU/src/ImageLoader.cu GPU/src/WindowFeatureComputer.cu GPU/src/ImageFeatureComputer.cu GPU/src/Utils.cpp GPU/src/CudaFunctions.cu GPU/src/FeatureComputer.cu GPU/src/GLCM.cu GPU/src/Image.cu GPU/src/Main.cpp GPU/src/ImageData.cu GPU/src/ProgramArguments.cpp GPU/src/Window.cu GPU/src/WorkArea.cu -o HaraliCU -O3 -Xptxas -O3 -std=c++11 -use_fast_math -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lcudadevrt -rdc=true \
-gencode arch=compute_35,code=compute_35 \
-gencode arch=compute_37,code=compute_37 \
-gencode arch=compute_50,code=compute_50 \
-gencode arch=compute_52,code=compute_52 \
-gencode arch=compute_60,code=compute_60 \
-gencode arch=compute_61,code=compute_61
