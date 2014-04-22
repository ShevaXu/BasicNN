#ifndef _SIMPLE_NN_H
#define _SIMPLE_NN_H

#include <opencv2/core/core.hpp>

#include "matreader/matreader.hpp"

using namespace cv;

struct NNParams 
{
	double lambda,
		alpha;
	double momentum;
};

struct NNMats
{
	Mat X, Xw1,
		Y, YSoft,
		hidW1,
		t1, grad1,
		t2, grad2;
		/*t1, t1bias, grad1,
		t2, t2bias, grad2;*/
};

class Simple3LayerNN
{
public:
	Simple3LayerNN(int L_input, int L_hidden, int L_output) : m_nIn(L_input), m_nHid(L_hidden), m_nOut(L_output)
	{
		m_rng = new RNG();
		m_reader = new MatFileReader();
	}

	void loadTrainData(const char* filename)
	{
		m_reader->loadMatFile(filename);
		m_train.X = m_reader->getVariable("X");
		m_train.Y = m_reader->getVariable("y");
		// setting
		m_imgH = 20;
		m_imgW = 20;
		assert(m_imgH * m_imgW == m_train.X.cols);
		assert(m_nIn == m_train.X.cols);
		m_nTrain = m_train.X.rows;
		//
		m_train.Xw1 = Mat::ones(m_nTrain, m_nIn + 1, CV_64FC1);
		m_train.X.copyTo(m_train.Xw1(Range::all(), Range(1, m_nIn + 1)));
		m_train.hidW1 = Mat::ones(m_nTrain, m_nHid + 1, CV_64FC1);
		//
		m_train.YSoft = Mat::zeros(m_nTrain, m_nOut, CV_64FC1);
		for (int i = 0; i < m_nTrain; i++)
		{
			//int label = (int)(m_trainY.at<double>(i, 0)) % m_nOut;	// map 10 to 0 back
			int label = (int)(m_train.Y.at<double>(i, 0)) - 1;	// map 10 to 9, 1 to 0
			m_train.YSoft.at<double>(i, label) = 1.0;
		}
	}

	void loadWeights(const char* filename)
	{
		m_reader->loadMatFile(filename);
		m_train.t1 = m_reader->getVariable("Theta1");
		m_train.t2 = m_reader->getVariable("Theta2");
		//
		//m_train.t1bias = m_train.t1.colRange(1, m_train.t1.cols);
		//m_train.t2bias = m_train.t2.colRange(1, m_train.t2.cols);
	}

	void loadTestData(const char* filename)
	{
		m_reader->loadMatFile(filename);
		m_test.X = m_reader->getVariable("X");
		// for test, y may not exist, return Mat()
		m_train.Y = m_reader->getVariable("y");
		//
		m_nTest = m_test.X.rows;
	}

	void setParams(const NNParams &params)
	{
		m_params = params;
	}

	void displayData(double factor = 1.0);

	void displayWeight(double factor = 2.0);

	void train();

	void predict();

	double evaluate(NNMats &mats);

private:
	int m_nIn,
		m_nHid,
		m_nOut;

	int m_imgW,
		m_imgH;

	int m_nTrain,
		m_nTest;

	NNMats m_train,
		m_test,
		m_check;

	/*Mat m_X4check,
		m_y4check,
		m_ySoft4check,
		m_hid4check,
		m_theta1c,
		m_theta2c;

	Mat m_trainX,
		m_testX,		
		m_trainY,
		m_testY;

	Mat m_trainXw1,
		m_testXw1,
		m_hidLw1,
		m_trainYSoft;

	Mat m_theta1,
		m_theta2,
		m_theta1debug,
		m_theta2debug,
		m_theta1NoBias,
		m_theta2NoBias;

	Mat m_gradTheta1,
		m_gradTheta2;*/

	RNG *m_rng,
		m_disRng;
	MatFileReader *m_reader;

	NNParams m_params;

protected:
	/*void sigmoid(const Mat &src, Mat &dst)
	{
		Mat temp;
		cv::exp(-src, temp);
		dst = 1.0 / (Scalar::all(1.0) + temp);
	}*/

	Mat sigmoid(const Mat &src)
	{
		Mat dst, temp;
		cv::exp(-src, temp);
		dst = 1.0 / (Scalar::all(1.0) + temp);
		return dst;
	}

	/*void sigmoidGradient(const Mat &src, Mat &dst)
	{
		Mat temp;
		sigmoid(src, temp);
		dst = temp.mul(Scalar::all(1.0) - temp);
	}*/

	Mat sigmoidGradient(const Mat &src)
	{
		Mat dst, 
			temp = sigmoid(src);
		dst = temp.mul(Scalar::all(1.0) - temp);
		return dst;
	}

	Mat initWeight(int L_in, int L_out, double factor)
	{
		Mat weight(L_out, L_in + 1, CV_64FC1);
		m_rng->fill(weight, RNG::UNIFORM, Scalar(-factor), Scalar(factor));
		return weight;
	}

	// for checking
	//double getCost(NNMats &mats, bool regularized = true);
	double getCost(NNMats &mats, const Mat &offset1, const Mat &offset2, bool regularized = true);

	//void getGradients(Mat &grad1, Mat &grad2, bool regularized = true);

	// for actual training
	double getCostAndGradients(NNMats &mats, bool regularized = true);

	void updateWeights(NNMats &mats);

	void computeNumericalGradients(Mat &grad1, Mat &grad2, bool regularized = true);

	bool gradientsCheck();
};

#endif // simpleNN.h