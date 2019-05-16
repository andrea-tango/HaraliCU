
#include "Utils.h"

/* Support method for saving the results in the right output folder */
void Utils::createFolder(string folderPath){
    if (mkdir(folderPath.c_str(), 0777) == -1) {
        if (errno == EEXIST) {
            // Alredy exists
        } else {
            // Something else
            cerr << "Cannot create the output folder: " << folderPath << endl
                 << "Error:" << strerror(errno) << endl;
        }
    }
}


// Removing the path and keeping filename+extension
string Utils::basename( std::string const& pathname ){
    return string(
            find_if( pathname.rbegin(), pathname.rend(),
                     MatchPathSeparator() ).base(),
            pathname.end() );
}

// Removing the extension from the filename
string Utils::removeExtension( std::string const& filename ){
    string::const_reverse_iterator
            pivot
            = find( filename.rbegin(), filename.rend(), '.' );
    return pivot == filename.rend()
           ? filename
           : std::string( filename.begin(), pivot.base() - 1 );
}