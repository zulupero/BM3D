#ifndef IMGHELPER_H
#include "opencv2/core/core.hpp"
using namespace cv;
#endif // IMGHELPER_H

class ImgHelper
{
    public:
        ~ImgHelper()
        {}

        static void transform2D(Mat* image);
        static void transform2DCuda(Mat* image);

    private:
        ImgHelper();

};
