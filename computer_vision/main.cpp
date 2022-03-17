#include <cstdio>
#include <opencv2/opencv.hpp>

int main() {

    struct CallbackData {
        cv::Mat canvas = cv::Mat::ones({ 500, 500 }, CV_8UC3);
        int radius = 3;
        cv::Scalar color = { 0.0, 255.0, 0.0, 255.0 };
    } callbackData;
    callbackData.canvas = cv::Scalar(255.0, 255.0, 255.0);

    cv::namedWindow("Canvas");
    auto mouseCallback = [](const int event, const int x, const int y, const int flags, void* userData) {
        auto c = *(CallbackData*)(userData);

        switch(event) {
            case cv::EVENT_LBUTTONDOWN: {
                cv::circle(c.canvas, { x, y }, c.radius, c.color, cv::FILLED);
            }
            break;
            case cv::EVENT_MOUSEMOVE: {
                if(flags & cv::EVENT_FLAG_LBUTTON) {
                    cv::circle(c.canvas, { x, y }, c.radius, c.color, cv::FILLED);
                }
            }
            break;
            default: {} break;
        }
    };
    cv::setMouseCallback("Canvas", mouseCallback, &callbackData);

    while(true) {
        cv::imshow("Canvas", callbackData.canvas);

        const int ch = cv::waitKey(1);
        if(ch == 27 || !static_cast<bool>(cv::getWindowProperty("Canvas", cv::WindowPropertyFlags::WND_PROP_VISIBLE))) {
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}
