#define STEREO 1

#if !STEREO
    #include "camera_calibration.cpp"
#else
    #include "stereo_calibration.cpp"
#endif
