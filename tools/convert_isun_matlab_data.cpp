// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_isun_data [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files, in the format as subfolder1/file1.JPEG
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include "matio.h"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");

void CVMatShow(const std::string& winname, cv::Mat img_in){
    double maxVal = 0.0;
    cv::minMaxIdx(img_in, NULL, &maxVal);
    if(maxVal == 0.0)
    {
        LOG(ERROR) << ": maxVal is 0 ";
        return;
    }
    cv::Mat img_8UC1;
    img_in.convertTo(img_8UC1, CV_8U, 255.0/maxVal);
    cv::Mat img_color;
    cv::applyColorMap(img_8UC1, img_color, cv::COLORMAP_JET);
    imshow(winname, img_color);
    cv::waitKey(0);
}

cv::Mat ReadMatlabMatToCVMat(const std::string& filename) {
    if(filename.empty())
    {
        LOG(ERROR) << ": File name is empty.";
        return cv::Mat();
    }

    mat_t  *mat;
    matvar_t *matvar;
    mat = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
    if(NULL == mat)
    {
        LOG(ERROR) << ": Can not open file.";
        return cv::Mat();
    }

    matvar = Mat_VarReadInfo(mat, "I");
    if(NULL == matvar)
    {
        Mat_Close(mat);
        LOG(ERROR) << ": Can not read information about I.";
        return cv::Mat();
    }

    if(matvar->class_type == MAT_C_DOUBLE && matvar->rank == 2)
    {
        int start[2] = {0, 0}, stride[2] = {1, 1}, edge[2] = {0, 0};
        edge[0] = matvar->dims[0];
        edge[1] = matvar->dims[1];
        int datasize = matvar->dims[0] * matvar->dims[1];

        double* data = new double[datasize]();
        Mat_VarReadData(mat, matvar, data, start, stride, edge);
        cv::Mat cv_mat = cv::Mat(matvar->dims[1], matvar->dims[0], CV_64F, data).t();
        cv::Mat mat_clone = cv_mat.clone();
        delete[] data;
        data = NULL;
        Mat_VarFree(matvar);
        Mat_Close(mat);
        return mat_clone;
    }
    else
    {
        Mat_VarFree(matvar);
        Mat_Close(mat);
        LOG(ERROR) << ": I is not satisfied with the conditions.";
        return cv::Mat();
    }
}

cv::Mat ReadMatlabMatToCVMat(const std::string& filename, const int height,
    const int width) {
    cv::Mat cv_mat = ReadMatlabMatToCVMat(filename);
    if (cv_mat.empty()) {
        LOG(ERROR) << ": Could not open or find file " << filename;
        return cv_mat;
    }

    cv::Mat cv_img;
    cv::Mat cv_flatimg;
    if (height > 0 && width > 0) {
        cv::resize(cv_mat, cv_img, cv::Size(width, height));
        if(cv_img.isContinuous())
        {
            cv_flatimg = cv_img.reshape(0, 1);
        }
    } else {
        cv_img = cv_mat;
        if(cv_img.isContinuous())
        {
            cv_flatimg = cv_img.reshape(0, 1);
        }
    }
    return cv_flatimg;
}

bool isunReadImageToDatum(const std::string& filename, const int height,
    const int width, Datum* datum) {
    cv::Mat cv_img = ReadMatlabMatToCVMat(filename, height, width);
    if (cv_img.data) {
        CVMatToDatum(cv_img, datum);
        return true;
    } else {
        return false;
    }
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of matlab mat files to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_isun_matlab_data [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The iSun dataset for the training demo is at\n"
        "    http://lsun.cs.princeton.edu/\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/isun_saliency_prediction/convert_isun_matlab_data");
    return 1;
  }

  const bool check_size = FLAGS_check_size;

  std::ifstream infile(argv[2]);
  std::vector< std::string > lines;
  std::string filename;
  while (infile >> filename) {
    lines.push_back(filename);
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size = 0;
  bool data_size_initialized = false;

  for (unsigned int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;

    status = isunReadImageToDatum(root_folder + lines[line_id],
        resize_height, resize_width, &datum);
    if (status == false) continue;
    if (check_size) {
      if (!data_size_initialized) {
        data_size = datum.channels() * datum.height() * datum.width();
        data_size_initialized = true;
      } else {
        const std::string& data = datum.data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    // sequential
    int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
        lines[line_id].c_str());

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(string(key_cstr, length), out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(ERROR) << "Processed " << count << " files.";
  }

  return 0;
}
