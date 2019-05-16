/**
 *                                     HaraliCU - GPU version                                                                  
 * Haralick feature extraction on medical images exploiting the full dynamics of gray-scale levels
 *
 * Copyright (C) 2019 Leonardo Rundo & Andrea Tangherloni
 *
 * Distributed under the terms of the GNU General Public License (GPL)
 *
 * This file is part of HaraliCU.
 *
 * HaraliCU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License v3.0 as published by
 * the Free Software Foundation.
 * 
 * HaraliCU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the
 * GNU General Public License for more details.
 *
 **/

#include <iostream>
#include <assert.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <chrono> // Performance monitor
#include "ImageFeatureComputer.h"

using namespace std;
using namespace cv;
using namespace chrono;


int main(int argc, char* argv[])
{
    cout << "********************************************* HaraliCU *********************************************" << endl << endl;

    ProgramArguments pa = ProgramArguments::checkOptions(argc, argv);

    typedef high_resolution_clock Clock;
    Clock::time_point t1 = high_resolution_clock::now();

    // Launching the external component
    ImageFeatureComputer ifc(pa);
    ifc.compute();

    // Execution time
    Clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    cout << endl << endl << "* Processing took " << time_span.count() << " seconds." << endl;

    cout << endl << "****************************************************************************************************" << endl;

	return 0;
}
