#include <iostream>
#include <cstring>
#include "Direction.h"

Direction::Direction(int directionNumber) {
    switch (directionNumber){
        case 1:{
            char templabel[20] = "Direction 0°";
            memcpy(this->label, templabel, 20);
            this->label[20] = 0;
            shiftRows = 0;
            shiftColumns = 1;
            break;
        }
        case 2:{
            char templabel[20] = "Direction 45°";
            memcpy(this->label, templabel, 20);
            this->label[20] = 0;
            shiftRows = -1;
            shiftColumns = 1;
            break;
        }
        case 3:{
            char templabel[20] = "Direction 90°";
            memcpy(this->label, templabel, 20);
            this->label[20] = 0;
            shiftRows = -1;
            shiftColumns = 0;
            break;
        }
        case 4:{
            char templabel[20] = "Direction 135°";
            memcpy(this->label, templabel, 20);
            this->label[20] = 0;
            shiftRows = -1;
            shiftColumns = -1;
            break;
        }
        default:
            fprintf(stderr, "Unsupported direction");
            exit(-1);
    }
}

void Direction::printDirectionLabel(const int direction){
    switch(direction){
        case 1:
            printf(" * Direction 0° *\n");
        case 2:
            printf(" * Direction 45° *\n");
        case 3:
            printf(" * Direction 90° *\n");
        case 4:
            printf(" * Direction 135° *\n");
        default:
            fprintf(stderr, "Fatal Error! Unsupported direction");
            exit(-1);
    }
}

