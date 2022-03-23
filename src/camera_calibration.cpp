#include <cstdio>
#include <opencv2/opencv.hpp>

using ImagePoints = std::vector<std::vector<cv::Point2f>>;
using ObjectPoints = std::vector<std::vector<cv::Point3f>>;

const int   g_camIndex = 5, g_camDelay = 100;
const int   g_numFrames = 25, g_winSize = 11;

const float g_squareSize = 30, g_aspectRatio = 1;
const auto  g_boardSize = cv::Size(7, 7);
const int   g_flags = cv::CALIB_FIX_ASPECT_RATIO;

struct ReprojectionErrors {
    double totalAverageError = 0.0;
    std::vector<float> perViewErrors;
};

static ReprojectionErrors computeReprojectionErrors(
    const ImagePoints &imagePoints, const ObjectPoints &objectPoints,
    const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
    const std::vector<cv::Mat> &rvecs, const std::vector<cv::Mat> &tvecs
) {
    using namespace cv;
    ReprojectionErrors ret;

    std::vector<Point2f> imagePoints2;
    size_t totalPoints = 0;
    double totalErr = 0, err;
    ret.perViewErrors.resize(objectPoints.size());

    for(size_t i = 0; i < objectPoints.size(); ++i) {
        cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
        err = cv::norm(imagePoints[i], imagePoints2, NORM_L2);

        size_t n = objectPoints[i].size();
        ret.perViewErrors[i] = (float)std::sqrt(err * err / n);
        totalErr += err * err;
        totalPoints += n;
    }

    ret.totalAverageError = std::sqrt(totalErr / totalPoints);
    return ret;
}

struct CalibrationResult {
    bool ok = false;
    cv::Mat cameraMatrix, distCoeffs;
};

static CalibrationResult runCalibration(cv::Size imageSize, int flags, ImagePoints imagePoints, float gridWidth/* , bool releaseObject */) {
    using namespace cv;
    CalibrationResult ret;

    ret.cameraMatrix = Mat::eye(3, 3, CV_64F);
    ret.distCoeffs = Mat::zeros(8, 1, CV_64F);
    if(flags & CALIB_FIX_ASPECT_RATIO) {
        ret.cameraMatrix.at<double>(0, 0) = g_aspectRatio;
    }

    ObjectPoints objectPoints(1);

    for(int i = 0; i < g_boardSize.height; ++i) {
        for(int j = 0; j < g_boardSize.width; ++j) {
            objectPoints[0].push_back({ j * g_squareSize, i * g_squareSize, 0 });
        }
    }

    objectPoints[0][g_boardSize.width - 1].x = objectPoints[0][0].x + gridWidth;
    objectPoints.resize(imagePoints.size(), objectPoints[0]);
    auto newObjPoints = objectPoints[0];

    //Find intrinsic and extrinsic camera parameters
    int iFixedPoint = -1;
    std::vector<cv::Mat> rvecs, tvecs;
    auto rms = calibrateCameraRO(
        objectPoints, imagePoints, imageSize, iFixedPoint,
        ret.cameraMatrix, ret.distCoeffs, rvecs, tvecs, newObjPoints,
        flags | CALIB_USE_LU
    );

    printf("Re-projection error reported by calibrateCamera: %.6f\n", rms);

    ret.ok = checkRange(ret.cameraMatrix) && checkRange(ret.distCoeffs);

    objectPoints.clear();
    objectPoints.resize(imagePoints.size(), newObjPoints);
    auto reprojErrors = computeReprojectionErrors(imagePoints, objectPoints, ret.cameraMatrix, ret.distCoeffs, rvecs, tvecs);

    return ret;
}

enum class ProgramMode { Detection = 0, Capturing, Calibrated };

int main() {
    using namespace cv;

    auto mode = ProgramMode::Detection;
    bool showUndistorted = false;

    Mat view;
    ImagePoints imagePoints;
    Size imageSize;
    CalibrationResult calib;
    VideoCapture capture; capture.open(g_camIndex);

    if(!capture.isOpened()) {
        printf("Unable to open camera at index %d!", g_camIndex);
        return -1;
    }

    float gridWidth = g_squareSize * (g_boardSize.width - 1);

    cv::namedWindow("Canvas", WindowFlags::WINDOW_NORMAL);
    clock_t prevTime = clock();
    while(true) {
        capture.read(view);
        imageSize = view.size();

        if(mode == ProgramMode::Capturing) {
            if(imagePoints.size() >= g_numFrames) {
                calib = runCalibration(imageSize, g_flags, imagePoints, gridWidth);
                mode = calib.ok ? ProgramMode::Calibrated : ProgramMode::Detection;
            }
        }

        int chessboardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
        std::vector<Point2f> corners;
        if(findChessboardCorners(view, g_boardSize, corners, chessboardFlags)) {
            Mat viewGray;
            cv::cvtColor(view, viewGray, COLOR_BGR2GRAY);
            cv::cornerSubPix(viewGray, corners, { g_winSize, g_winSize }, { -1, -1 }, { TermCriteria::EPS | TermCriteria::COUNT, 30, 0.0001 });

            // For camera only take new samples after delay time
            if(mode == ProgramMode::Capturing && (clock() - prevTime > g_camDelay * 1e-3 * CLOCKS_PER_SEC)) {
                imagePoints.push_back(corners);
                prevTime = clock();
            }

            drawChessboardCorners(view, g_boardSize, Mat(corners), true);
        }

        if(mode == ProgramMode::Calibrated && showUndistorted) {
            Mat temp = view.clone();
            undistort(temp, view, calib.cameraMatrix, calib.distCoeffs);
        }

        if(view.empty()) {
            view = Mat::zeros(imageSize, view.type());
        }

        int baseLine = 0;

        std::string msg;
        switch(mode) {
            case ProgramMode::Detection: {
                msg = "Press 'g' to start";
            } break;

            case ProgramMode::Capturing: {
                msg = cv::format(showUndistorted ? "%d/%d Undist" : "%d/%d", (int)imagePoints.size(), g_numFrames);
            } break;

            case ProgramMode::Calibrated: {
                msg = "Calibrated";
            } break;
        }

        auto textSize = getTextSize(msg, 1, 1, 1, &baseLine);
        Point textOrigin(view.cols - 2 * textSize.width - 10, view.rows - 2 * baseLine - 10);
        cv::putText(view, msg, textOrigin, cv::FONT_HERSHEY_COMPLEX, 1, mode == ProgramMode::Calibrated ? Scalar { 0, 255, 0 } : Scalar { 0, 0, 255 });
        cv::imshow("Canvas", view);

        const int key = cv::waitKey(1);
        switch(key) {
            case 'u': {
                if(mode == ProgramMode::Calibrated) {
                    showUndistorted = !showUndistorted;
                }
            } break;

            case 'g': {
                mode = ProgramMode::Capturing;
                imagePoints.clear();
            }

            default: {} break;
        }

        if(key == 27 || !static_cast<bool>(cv::getWindowProperty("Canvas", WindowPropertyFlags::WND_PROP_VISIBLE))) {
            break;
        }
    }

    capture.release();
    cv::destroyAllWindows();

    return 0;
}
