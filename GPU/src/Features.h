#ifndef FEATURES_H_
#define FEATURES_H_

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__ 
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

#include <vector>
#include <string>
#include <iostream>

using namespace std;

/**
 * List of all the supported features.
 * The index in the enumeration is used for accessing the right cell
 * when saving the results in a feature array
 */
enum FeatureNames {
    ASM,
    AUTOCORRELATION,
    ENTROPY,
    MAXPROB,
    HOMOGENEITY,
    CONTRAST,
    CORRELATION,
    CLUSTERPROMINENCE,
    CLUSTERSHADE,
    SUMOFSQUARES,
    DISSIMILARITY,
    IDM,
    // Sum Aggregated
    SUMAVERAGE,
    SUMENTROPY,
    SUMVARIANCE,
    // Diff Aggregated
    DIFFENTROPY,
    DIFFVARIANCE,
    // Marginal probability feature
    IMOC
};

/**
 * Helper class that lists all supported features and offers some
 * utilities methods about them
 */
class Features {
public:
     /**
     * Returns a list of all the supported features
     * @return list of all the supported features
     */
	CUDA_HOSTDEV static int getSupportedFeaturesCount();
    /**
     * Returns a list of all the file names associated with the features
     * @return list of all the file names associated with the features
     */
	CUDA_HOST static void printFeatureName(FeatureNames featureName);
	/**
     * The number of features supported by HaraliCU; used for allocating
     * arrays of features
     * @return number of features supported by this tool
     */
	CUDA_HOST static vector<FeatureNames> getAllSupportedFeatures();
	/**
     * Returns a list of all the file names associated with the features
     * @return list of all the file names associated with the features
     */
    CUDA_HOST static vector<string> getAllFeaturesFileNames();
	/**
     * DEBUG METHOD. This method prints the feature labels and their values
     * @param features
     */
    CUDA_HOST static void printAllFeatures(const vector<double>& features);
	/**
     * DEBUG METHOD. This method prints single feature label
     * @param features list of computed features
     * @param featureName index of the feature in the enumeration
     */
    CUDA_HOST static void printSingleFeature(const vector<double> &features,
	                                   FeatureNames featureName);
    /**
     * DEBUG METHOD. This method prints single feature label and its value
     * @param value: value of the feature to print
     * @param fname: index of the feature in the enumeration
     * @return
     */
    CUDA_HOST static string printFeatureNameAndValue(double value, FeatureNames fname);

	/**
     * Returns (as a string) the label associated with the enum
     * @param featureName whose label will be returned
     */
    CUDA_HOST static string getFeatureName(FeatureNames featureName);

};

#endif /* FEATURES_H_ */
