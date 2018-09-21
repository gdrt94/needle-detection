#include <opencv2/opencv.hpp>
#include <random>
#include <iostream>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

using namespace cv;
using namespace std;

void visualise (const vector<Point2f>& needlePoints);

int main()
{

vector<Point2f> boundingBoxPoints;

// -----Working on B-scans starts here-----
// decrease 69 for testing
for(int index = 18; index < 25; index++) {

    // Note: at 18 appears (at 17 needle cluster isn't even created), at 71 direction changes
    // int index = 67;

    // check clustering at 18,70,71,72,107-110
    // TODO too many points in the retina, which reduces the chances of point selection in the needle part...
    // TODO ...possible solution assign the point with highest Y coord to be one of the centers

   // RANSAC(no_update_steps = 1000, deviation_pixels = 2/3) provides good results from 26 - 67 (ex.65)

    String indexStr = to_string(index);
    if (index < 10)
        indexStr = "00" + indexStr;
    else if (index < 100)
        indexStr = "0" + indexStr;

    // Reading
    String path = "/home/gudrat/Documents/cluster-and-fit/img/" + indexStr + ".bmp";
    Mat colorImage = imread(path);

    // Preprocessing
    cvtColor(colorImage, colorImage, COLOR_BGR2GRAY);
    Mat noise = Mat(colorImage.size(), colorImage.type());

    Scalar a(0);
    Scalar b(0.0);
    int kernelg = 3;
    int kernel = 1;
    double thresh = 80;

    randn(noise, a, b);
    colorImage += noise;

    GaussianBlur(colorImage, colorImage, Size(kernelg, kernelg), 0, 0);
    medianBlur(colorImage, colorImage, kernel);
    cvtColor(colorImage, colorImage, COLOR_GRAY2BGR);

    // Initializing clustering
    vector<Point2f> clusterPoints;

    // Highlighting points of interest
    for (int x = 0; x < colorImage.cols; ++x) {
        for (int y = 0; y < colorImage.rows; ++y) {

            Vec3b colorPoint = colorImage.at<Vec3b>(Point2d(x, y));
            // Deleting dark points
            if (colorPoint.val[0] < thresh && colorPoint.val[1] < thresh && colorPoint.val[2] < thresh) {
                colorPoint.val[0] = 255;
                colorPoint.val[1] = 255;
                colorPoint.val[2] = 255;
            } else { // Converting white points to red and adding to clusterPoints
                colorPoint.val[0] = 0;
                colorPoint.val[1] = 0;
                colorPoint.val[2] = 255;
                clusterPoints.emplace_back(x, y);
            }
            colorImage.at<Vec3b>(Point2d(x, y)) = colorPoint;
        }
    }

    // Displaying points of interest
    namedWindow("Color image " + indexStr, WINDOW_NORMAL);
    resizeWindow("Color image " + indexStr, 384, 768);
    imshow("Color image " + indexStr, colorImage);

    // Performing clustering
    Mat centers;
    vector<int> labels(clusterPoints.size());
    for (int i = 0; i < labels.size(); i++) {
        if (clusterPoints[i].y < 0.6*colorImage.rows &&
            clusterPoints[i].x > 0.33*colorImage.cols &&
            clusterPoints[i].x < 0.67*colorImage.cols)
            labels[i] = 0;
        else if (clusterPoints[i].x < 0.5*colorImage.cols)
            labels[i] = 1;
        else
            labels[i] = 2;
    }

    int clusterCount = 3;
    kmeans(clusterPoints, clusterCount, labels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
           5, KMEANS_USE_INITIAL_LABELS, centers); // TODO play with attempts

    // Drawing clusters
    colorImage = Scalar::all(0);
    Scalar colorTab[] =
    {
            Scalar(0, 0, 255),
            Scalar(0,255,0),
            Scalar(255,100,100),
            Scalar(192,192,192),
    };
    int pointsPerClass[3] = {0};
    for(int i = 0; i < clusterPoints.size(); i++)
    {
        int clusterIdx = labels[i]; // labels.at<int>(i); commented code is obsolete; I used it when "labels" was of type Mat
        pointsPerClass[labels[i]]++;
        Point ipt = clusterPoints[i];
        circle(colorImage, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA );
    }
    namedWindow("Clusters " + indexStr, WINDOW_NORMAL);
    resizeWindow("Clusters " + indexStr, 384, 768);
    imshow("Clusters " + indexStr, colorImage);

    // Selecting ellipse cluster; TODO currently simply defined, later make the ellipse cluster the one which is in center
    int ellipseClusterIndex = -1;
    if (pointsPerClass[0] > 0 && pointsPerClass[1] > 0 && pointsPerClass[2] > 0) {
        ellipseClusterIndex = 0;
        if (pointsPerClass[1] < pointsPerClass[ellipseClusterIndex])
            ellipseClusterIndex = 1;
        if (pointsPerClass[2] < pointsPerClass[ellipseClusterIndex])
            ellipseClusterIndex = 2;
    }

    // Showing ellipse cluster if there is
    vector<Point> ellipsePoints;
    if (ellipseClusterIndex > -1) {
        colorImage = Scalar::all(0);
        for(int i = 0; i < clusterPoints.size(); i++)
        {
            if (labels[i] == ellipseClusterIndex) {
                int clusterIdx = ellipseClusterIndex;
                Point ipt = clusterPoints[i];
                // For numerical accuracy defining the image center as the origin, so that when multiplying x and y, we don't get too big numbers
                ellipsePoints.emplace_back(ipt.x - 256, ipt.y - 512);
                circle(colorImage, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA);
            }
        }
        namedWindow("Ellipse cluster " + indexStr, WINDOW_NORMAL);
        resizeWindow("Ellipse cluster " + indexStr, 384, 768);
        imshow("Ellipse cluster " + indexStr, colorImage);
    } else {
        return EXIT_FAILURE;
    }

    // -----------RANSAC-----------

    Mat bestCandidate;
    int mostNumberOfClosePoints = -1;
    int stepsWithoutUpdates = 0;
    vector<Point> bestRandom5Points;
    vector<int> bestInliers;
    // Assuming that an ellipse is approximately 90x90 pixels
    double f0 = 90.0;
    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<int> dist(0, static_cast<int>(ellipsePoints.size()));
    int steps_temp = 0; // this is for counting how many steps algorithm will make overall
    while (stepsWithoutUpdates < 1000) { // this number can be increased for better accuracy, but processing time will be increased

        steps_temp++;

        // Randomly selecting 5 points from the cluster
        vector<Vec6d> random5Points;
        vector<Point> normalRandom5Points;

        // Converting the points to ellipse representation form
        for (int i = 0; i < 5; i++) {
            Point randomPoint = ellipsePoints[dist(mt)];
            Vec6d pointInEllipseForm(1.0 * randomPoint.x * randomPoint.x,
                                     2.0 * randomPoint.x * randomPoint.y,
                                     1.0 * randomPoint.y * randomPoint.y,
                                     2.0 * f0 * randomPoint.x,
                                     2.0 * f0 * randomPoint.y,
                                     f0 * f0);
            random5Points.emplace_back(pointInEllipseForm);
            normalRandom5Points.emplace_back(randomPoint);
        }

        // Defining matrices
        Mat M0, M1, M2, M3, M4, M0T, M1T, M2T, M3T, M4T;

        // Converting point vectors to matrices
        M0 = Mat(random5Points[0]);
        M1 = Mat(random5Points[1]);
        M2 = Mat(random5Points[2]);
        M3 = Mat(random5Points[3]);
        M4 = Mat(random5Points[4]);

        // Transposing matrices
        transpose(M0, M0T);
        transpose(M1, M1T);
        transpose(M2, M2T);
        transpose(M3, M3T);
        transpose(M4, M4T);

        // Calculating the symmetric matrix
        Mat resultingMat = M0 * M0T + M1 * M1T + M2 * M2T + M3 * M3T + M4 * M4T;

        // Computing the unit eigenvector θ of the symmetric resultingMat matrix for the smallest eigenvalue, and storing it as a candidate
        // Because computing the unit eigenvector for the smallest eigenvalue of a symmetric matrix M...
        // ...is equivalent to computing the unit vector θ that minimizes the quadratic form (cost function)
        Mat eigenvalues, eigenvectors;
        eigen(resultingMat, eigenvalues, eigenvectors);
        Mat candidate = eigenvectors.row(5); // this can be normalized; do I need to take into account only non-zero eigenvalues?
        transpose(candidate, candidate);

        // Finding the number of close points to candidate ellipse using covariance matrix
        int numberOfClosePoints = 0;
        vector<int> inliers(ellipsePoints.size(), 0);
        for (int i = 0; i < ellipsePoints.size(); ++i) {

            // Converting the point in ellipse cluster to ellipse representation form
            Vec6d pointInEllipseForm(1.0 * ellipsePoints[i].x * ellipsePoints[i].x,
                                     2.0 * ellipsePoints[i].x * ellipsePoints[i].y,
                                     1.0 * ellipsePoints[i].y * ellipsePoints[i].y,
                                     2.0 * f0 * ellipsePoints[i].x,
                                     2.0 * f0 * ellipsePoints[i].y,
                                     f0 * f0);

            // Defining the covariance matrix of this point
            Mat covarianceMat(6, 6, CV_64F, 0.0);
            covarianceMat.at<double>(0, 0) = 4.0 * ellipsePoints[i].x * ellipsePoints[i].x;

            covarianceMat.at<double>(2, 2) = 4.0 * ellipsePoints[i].y * ellipsePoints[i].y;

            covarianceMat.at<double>(3, 3) =
            covarianceMat.at<double>(4, 4) = 4.0 * f0 * f0;

            covarianceMat.at<double>(0, 1) =
            covarianceMat.at<double>(1, 0) =
            covarianceMat.at<double>(1, 2) =
            covarianceMat.at<double>(2, 1) = 4.0 * ellipsePoints[i].x * ellipsePoints[i].y;

            covarianceMat.at<double>(0, 3) =
            covarianceMat.at<double>(3, 0) =
            covarianceMat.at<double>(1, 4) =
            covarianceMat.at<double>(4, 1) = 4.0 * f0 * ellipsePoints[i].x;

            covarianceMat.at<double>(1, 3) =
            covarianceMat.at<double>(3, 1) =
            covarianceMat.at<double>(2, 4) =
            covarianceMat.at<double>(4, 2) = 4.0 * f0 * ellipsePoints[i].y;

            covarianceMat.at<double>(1, 1) = 4.0 * (ellipsePoints[i].x * ellipsePoints[i].x +
                                                    ellipsePoints[i].y * ellipsePoints[i].y);

            // Defining threshold for admissible deviation from the fitted ellipse (3 pixels)
            double deviationThreshold = 3.0 * 3.0;

            // [SKIP] Dot product using matrix multiplication
            // [SKIP] Mat tempMattt;
            // [SKIP] transpose(Mat(pointInEllipseForm), tempMattt);
            // [SKIP] tempMattt = tempMattt*candidate;

            // Computing the deviation of the selected point
            double nominator = Mat(pointInEllipseForm).dot(candidate);
            nominator = nominator * nominator;
            Mat covarianceTimesCandidate = covarianceMat * candidate;
            double denominator = candidate.dot(covarianceTimesCandidate);
            double deviation = nominator / denominator;

            if (deviation < deviationThreshold) {
                numberOfClosePoints++;
                inliers[i] = 1;
            }
        }

        // Updating number of close points and keeping track of how many times there was no update
        if (numberOfClosePoints > mostNumberOfClosePoints) {
            mostNumberOfClosePoints = numberOfClosePoints;
            bestCandidate = candidate;
            bestRandom5Points = normalRandom5Points;
            bestInliers = inliers;
            stepsWithoutUpdates = 0;
        } else {
            stepsWithoutUpdates++;
        }
    }

    // Highlighting the random points that formed best ellipse and removing outliers
    colorImage = Scalar::all(0);
    vector<Point> inlierPoints;
    for (int i = 0; i < ellipsePoints.size(); i++)
    {
        // Converting points back to their original coordinates
        if (bestInliers[i] == 1) {
            Point ipt(ellipsePoints[i].x + 256, ellipsePoints[i].y + 512);
            inlierPoints.emplace_back(ipt);
            circle(colorImage, ipt, 2, colorTab[0], FILLED, LINE_AA);
        }
    }
    for (auto &bestRandom5Point : bestRandom5Points) {
        Point ipt (bestRandom5Point.x + 256, bestRandom5Point.y + 512);
        circle(colorImage, ipt, 2, colorTab[1], FILLED, LINE_AA);
    }

    // ----- This section is not very useful -----
    // What can it be useful for? [BUG] rarely but still sometimes doesn't work (e.g. 038.bmp)
    // Calculating the center of the fitted ellipse using partial derivatives
    // https://math.stackexchange.com/questions/2096758/finding-the-center-of-an-ellipse
    // ax+by+f0*d=0
    // bx+cy+f0*e=0

    double A = bestCandidate.at<double>(0,0);
    double B = bestCandidate.at<double>(1,0);
    double C = bestCandidate.at<double>(2,0);
    double D = bestCandidate.at<double>(3,0);
    double E = bestCandidate.at<double>(4,0);
    double centerY = (f0 * (B * D / A - E))/(C - B * B / A);
    double centerX = - (B * centerY + f0 * D) / A;

    // Highlighting the ellipse center on the image
    Point ellipseCenter((int)round(centerX) + 256, (int)round(centerY) + 512);
    circle(colorImage, ellipseCenter, 2, colorTab[2], FILLED, LINE_AA);
    namedWindow("Highlighting after outlier removal " + indexStr, WINDOW_NORMAL);
    resizeWindow("Highlighting after outlier removal " + indexStr, 384, 768);
    imshow("Highlighting after outlier removal " + indexStr, colorImage);
    // ----- Non-useful section ends here -----


    // Finding rotated bounding box around inliers TODO can also show around cluster (don't forget to add 256/512) and then aggregate results with inlier method
    RotatedRect rotatedRect = minAreaRect(inlierPoints);
    // boxPoints(rect, resultingBox); another implementation
    Point2f vertices[4];
    rotatedRect.points(vertices);
    for (int i = 0; i < 4; i++)
        line(colorImage, vertices[i], vertices[(i+1)%4], Scalar(0,255,0), 2);

    // We can also draw a straight rectangle, which isn't of minimum area, but it isn't useful
    //Rect brect = rotatedRect.boundingRect();
    //rectangle(colorImage, brect, Scalar(255,0,0), 2);

    // Finding midpoint of the lower line segment of the bounding box and adding it to the result
    float verticesYvalues[4];
    for (int i = 0; i < 4; ++i) {
        verticesYvalues[i] = vertices[i].y;
    }
    const int float_iterator_size = sizeof(verticesYvalues) / sizeof(float);
    long biggestIndex = distance(verticesYvalues, max_element(verticesYvalues, verticesYvalues + float_iterator_size));
    verticesYvalues[biggestIndex] = -99999999.0f;
    long secondBiggestIndex = distance(verticesYvalues, max_element(verticesYvalues, verticesYvalues + float_iterator_size));

    Point2f midPointOfBBox((vertices[biggestIndex].x + vertices[secondBiggestIndex].x) / 2,
                           (vertices[biggestIndex].y + vertices[secondBiggestIndex].y) / 2);
    circle(colorImage, midPointOfBBox, 5, colorTab[3], FILLED, LINE_AA);
    boundingBoxPoints.emplace_back(midPointOfBBox);

    // Highlighting the bounding box around inlier points and indicating the low segment midpoint of the bounding box
    namedWindow("Bounding boxes " + indexStr, WINDOW_NORMAL);
    resizeWindow("Bounding boxes " + indexStr, 384, 768);
    imshow("Bounding boxes " + indexStr, colorImage);

    // Debug printing
    // cout << "'DEBUG' number of close points = " << mostNumberOfClosePoints << endl;
    // cout << "'DEBUG' overall steps = " << steps_temp << endl;
    // cout << "'DEBUG' best candidate = " << endl << " " << bestCandidate << endl;
    // cout << "'DEBUG' point " + indexStr + " " << midPointOfBBox.x << " " << midPointOfBBox.y << endl;

    waitKey(0);
    // destroying windows opened for previous bmp file; maybe cvReleaseImage should also be used, but unlikely
    destroyAllWindows();
}
// -----Working on B-scans starts here-----
    visualise(boundingBoxPoints);

    return EXIT_SUCCESS;
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    return (viewer);
}

unsigned int text_id = 0;
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
                            void* viewer_void)
{
    pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
    if (event.getKeySym () == "r" && event.keyDown ())
    {
        std::cout << "r was pressed => removing all text" << std::endl;

        char str[512];
        for (unsigned int i = 0; i < text_id; ++i)
        {
            sprintf (str, "text#%03d", i);
            viewer->removeShape (str);
        }
        text_id = 0;
    }
}

void mouseEventOccurred (const pcl::visualization::MouseEvent &event,
                         void* viewer_void)
{
    pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
    if (event.getButton () == pcl::visualization::MouseEvent::LeftButton &&
        event.getType () == pcl::visualization::MouseEvent::MouseButtonRelease)
    {
        std::cout << "Left mouse button released at position (" << event.getX () << ", " << event.getY () << ")" << std::endl;

        char str[512];
        sprintf (str, "text#%03d", text_id ++);
        viewer->addText ("clicked here", event.getX (), event.getY (), str);
    }
}

void visualise (const vector<Point2f>& needlePoints)
{
// ------------------------------------
    // -----Create example point cloud-----
    // ------------------------------------
    pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    std::cout << "Generating example point clouds.\n\n";
    // We're going to make an ellipse extruded along the z-axis. The colour for
    // the XYZRGB cloud will gradually go from red to green to blue.
    uint8_t r(255), g(15), b(15);
    for (float z(-1.0); z <= 1.0; z += 0.05)
    {
        for (float angle(0.0); angle <= 360.0; angle += 5.0)
        {
            pcl::PointXYZ basic_point;
            basic_point.x = 0.5 * cosf (pcl::deg2rad(angle));
            basic_point.y = sinf (pcl::deg2rad(angle));
            basic_point.z = z;
            basic_cloud_ptr->points.push_back(basic_point);

            pcl::PointXYZRGB point;
            point.x = basic_point.x;
            point.y = basic_point.y;
            point.z = basic_point.z;
            uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                            static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
            point.rgb = *reinterpret_cast<float*>(&rgb);
            point_cloud_ptr->points.push_back (point);
        }
        if (z < 0.0)
        {
            r -= 12;
            g += 12;
        }
        else
        {
            g -= 12;
            b += 12;
        }
    }
    basic_cloud_ptr->width = (int) basic_cloud_ptr->points.size ();
    basic_cloud_ptr->height = 1;
    point_cloud_ptr->width = (int) point_cloud_ptr->points.size ();
    point_cloud_ptr->height = 1;

    // ----------------------------------------------------------------
    // -----Calculate surface normals with a search radius of 0.05-----
    // ----------------------------------------------------------------
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setInputCloud (point_cloud_ptr);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
    ne.setSearchMethod (tree);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals1 (new pcl::PointCloud<pcl::Normal>);
    ne.setRadiusSearch (0.05);
    ne.compute (*cloud_normals1);

    // ---------------------------------------------------------------
    // -----Calculate surface normals with a search radius of 0.1-----
    // ---------------------------------------------------------------
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
    ne.setRadiusSearch (0.1);
    ne.compute (*cloud_normals2);

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    viewer = simpleVis(basic_cloud_ptr);

    //--------------------
    // -----Main loop-----
    //--------------------
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
}


// Unused
int old () {
    String path = "/home/gudrat/Documents/cluster-and-fit/img/035.bmp";
    Mat grayImage = imread(path, IMREAD_GRAYSCALE);
    Mat colorImage = imread(path, IMREAD_COLOR);

    if(grayImage.empty())
    {
        cout <<  "Smth is wrong with image path" << endl;
        return EXIT_FAILURE;
    }

    // Create a window for display.
    namedWindow("Gray image", WINDOW_AUTOSIZE);
    namedWindow("Color image", WINDOW_AUTOSIZE);

    // Display image
    imshow("Gray image", grayImage);
    imshow("Color image", colorImage);
    // Wait for a keystroke in the window
    waitKey(0);

    return EXIT_SUCCESS;
}

void printingStuff() {
    cout << "double:" << endl;
    cout << "min: " << numeric_limits<float>::min() << endl;
    cout << "max: " << numeric_limits<float>::max() << endl;
    cout << endl;
}