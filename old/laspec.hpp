//
// Created by miguelyermo on 5/3/20.
//

/*
* FILENAME :  laspec.h
* PROJECT  :  rule-based-classifier-cpp
* DESCRIPTION :
*
*
*
*
*
* AUTHOR :    Miguel Yermo        START DATE : 20:23 5/3/20
*
*/
// LAS 1.4 Specification http://www.asprs.org/a/society/committees/standards/LAS_1_4_r13.pdf

#ifndef CPP_LASPEC_H
#define CPP_LASPEC_H

// TODO: Enum
#define UNCLASSIFIED   0
#define UNKNOWN        1
#define GROUND         2
#define LOW_VEG        3
#define MED_VEG        4
#define HIGH_VEG       5
#define BUILDING       6
#define LOW_POINT      7
#define SMALL_BUILDING 8 // Reserved field
#define WATER          9
#define RAIL           10
#define ROAD           11
#define OVERLAP        12
#define WIRE           14
#define ELECTRIC_TOWER 15
#define BRIDGE_DECK    17
#define HIGH_NOISE     18
#define ROAD_CANDIDATE 19
#define FACADE         20 // Reserved field
#define FIREBREAK      64
#define PATH           65
#define SPURIOUS       66
#define GRASS          101
#define TREE           105
#define PARKING        111

#endif //CPP_LASPEC_H
