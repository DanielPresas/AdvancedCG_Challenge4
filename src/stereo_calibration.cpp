#include <cstdio>
#include <opencv2/opencv.hpp>

const int g_camIndex = 5;

int main() {

    using namespace cv;

    Mat frame;
    VideoCapture capture; capture.open(g_camIndex);

    if(!capture.isOpened()) {
        printf("Unable to open camera at index %d!", g_camIndex);
        return -1;
    }

    while(true) {
        capture.read(frame);

        if(frame.empty()) {
            continue;
        }

        cv::imshow("Canvas", frame);
        const int key = cv::waitKey(1);
        if(key == 27 || !static_cast<bool>(cv::getWindowProperty("Canvas", WindowPropertyFlags::WND_PROP_VISIBLE))) {
            break;
        }
    }

    return 0;
}
