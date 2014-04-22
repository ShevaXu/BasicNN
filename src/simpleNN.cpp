#include <opencv2/highgui/highgui.hpp>

#include <iostream>

#include "simpleNN.h"

void Simple3LayerNN::displayData( double factor /*= 1.0*/ )
{
	Mat img(200, 200, CV_64FC1);
	int i, j, k = 0;
	for (i = 0; i < 10; i++)
	{
		for (j = 0; j < 10; j++)
		{
			int idx = (unsigned)m_disRng % m_nTrain;
			Mat temp = m_train.X.row(idx).reshape(0, 20).t();
			temp.copyTo(img(Rect(20 * i, 20 * j, 20, 20)));
			k++;
		}
	}
	//
	normalize(img, img, 0.0, 1.0, NORM_MINMAX);
	//
	if (factor == 1.0)
	{
		imshow("Display", img);
		waitKey();
	}
	else
	{
		Mat copy;
		resize(img, copy, Size(), factor, factor);
		imshow("Display", copy);
		waitKey();
	}
}

void Simple3LayerNN::displayWeight( double factor /*= 2.0*/ )
{
	Mat img(100, 100, CV_64FC1);
	Mat b1 = m_train.t1.colRange(1, m_train.t1.cols);
	int i, j, k = 0;
	for (i = 0; i < 5; i++)
	{
		for (j = 0; j < 5; j++)
		{						
			Mat temp = b1.row(k).reshape(0, 20).t();
			temp.copyTo(img(Rect(20 * i, 20 * j, 20, 20)));
			k++;
		}
	}
	//
	normalize(img, img, 0.0, 1.0, NORM_MINMAX);
	//
	if (factor == 1.0)
	{
		imshow("Display", img);
		waitKey();
	}
	else
	{
		Mat copy;
		resize(img, copy, Size(), factor, factor);
		imshow("Display", copy);
		waitKey();
	}
}

void Simple3LayerNN::train()
{
	// gradient check
	if(!gradientsCheck())
		return;
	else
		std::cout << "Gradient checking passed!\n";

	//
	m_train.t1 = initWeight(m_nIn, m_nHid, 0.2);
	m_train.t2 = initWeight(m_nHid, m_nOut, 0.2);
	int iterations = 100;
	for (int i = 0; i < iterations; i++)
	{
		std::cout << "Iteration " << i + 1 << ": ";
		double J = getCostAndGradients(m_train);
		updateWeights(m_train);
		std::cout << "cost " << J << std::endl;
	}
	double acc = evaluate(m_train);
	std::cout << "Accuracy on training set: " << acc << std::endl;
	

	/*double J = getCostAndGradients(m_train);
	std::cout << J << std::endl;
	Mat offset1 = Mat::zeros(m_train.grad1.size(), CV_64FC1);
	Mat offset2 = Mat::zeros(m_train.grad2.size(), CV_64FC1);
	J = getCost(m_train, offset1, offset2);
	std::cout << J << std::endl;*/

	// 
}

double Simple3LayerNN::getCostAndGradients( NNMats &mats, bool regularized /*= true*/ )
{
	int m = mats.X.rows;
	//
	Mat z_2 = mats.Xw1 * mats.t1.t();	// n * 25
	Mat temp = sigmoid(z_2);
	temp.copyTo(mats.hidW1(Range::all(), Range(1, mats.hidW1.cols)));
	Mat z_3 = mats.hidW1 * mats.t2.t();	// n * 10
	Mat prediction = sigmoid(z_3);
	//
	Mat diff = cv::abs(prediction - (1.0 - mats.YSoft));
	Mat logDiff;
	cv::log(diff, logDiff);
	Scalar val = sum(logDiff);
	double J = -val[0] / m;
	
	// back-prop
	Mat delta_3 = prediction - mats.YSoft;
	Mat temp2 = delta_3 * mats.t2;
	Mat sigGrad = sigmoidGradient(z_2);
	Mat delta_2 = temp2(Range::all(), Range(1, temp2.cols)).mul(sigGrad);
	mats.grad1 = delta_2.t() * mats.Xw1 * (1.0 / m);
	mats.grad2 = delta_3.t() * mats.hidW1 * (1.0 / m);

	if (regularized)
	{
		Mat b1 = mats.t1.colRange(1, mats.t1.cols),
			b2 = mats.t2.colRange(1, mats.t2.cols);
		Scalar reg = sum(b1.mul(b1)) + sum(b2.mul(b2));
		J += m_params.lambda / (2 * m) * reg[0];
		//
		mats.grad1(Range::all(), Range(1, mats.grad1.cols)) += (m_params.lambda / m) * mats.t1(Range::all(), Range(1, mats.t1.cols));
		mats.grad2(Range::all(), Range(1, mats.grad2.cols)) += (m_params.lambda / m) * mats.t2(Range::all(), Range(1, mats.t2.cols));
	}

	//
	return J;
}

bool Simple3LayerNN::gradientsCheck()
{
	int m = 10, nLabels = 3,
		n = 10, nHid = 5;
	m_check.X = initWeight(n - 1, m, 1.0);	// X is 10 by 10
	m_check.Xw1 = Mat::ones(m, n + 1, CV_64FC1);	// Xw1 is 10 by 11
	m_check.X.copyTo(m_check.Xw1(Range::all(), Range(1, n + 1)));
	m_check.Y.create(m, 1, CV_64FC1);
	m_check.YSoft = Mat::zeros(m, nLabels, CV_64FC1);
	for (int i = 0; i < m; i++)
	{
		int label = i % nLabels;
		m_check.Y.at<double>(i, 0) = (double)(i + 1);
		m_check.YSoft.at<double>(i, label) = 1.0;
	}
	//
	m_check.hidW1 = Mat::ones(m, nHid + 1, CV_64FC1);
	//
	m_check.t1 = initWeight(n, nHid, 0.12);
	m_check.t2 = initWeight(nHid, nLabels, 0.12);

	//
	/*double J = getCostAndGradients(m_check, true);
	std::cout << J << std::endl;*/
	getCostAndGradients(m_check);

	//
	Mat numgrad1, numgrad2;
	computeNumericalGradients(numgrad1, numgrad2);
	std::cout << numgrad1.at<double>(1, 1) << std::endl;

	//
	std::cout << (abs(m_check.grad1 - numgrad1) < 1e-9) << std::endl;
	std::cout << (abs(m_check.grad2 - numgrad2) < 1e-9) << std::endl;

	Scalar diff1 = mean(m_check.grad1 - numgrad1),
		diff2 = mean(m_check.grad2 - numgrad2);
	std::cout << "mean diffs: " << abs(diff1[0]) << " " << abs(diff2[0]) << std::endl;
	if (abs(diff1[0]) < 1e-9 && abs(diff2[0]) < 1e-9)
	{		
		return true;
	}
	else
		return false;
}

void Simple3LayerNN::computeNumericalGradients( Mat &grad1, Mat &grad2, bool regularized /*= true*/ )
{
	double eps = 1e-4;
	grad1 = Mat::zeros(m_check.grad1.size(), CV_64FC1);
	grad2 = Mat::zeros(m_check.grad2.size(), CV_64FC1);
	Mat offset1 = Mat::zeros(m_check.grad1.size(), CV_64FC1);
	Mat offset2 = Mat::zeros(m_check.grad2.size(), CV_64FC1);
	//
	for (int i = 0; i < offset1.rows; i++)
	{
		for (int j = 0; j < offset1.cols; j++)
		{
			Mat temp = offset1.clone();
			temp.at<double>(i, j) = eps;
			double J = getCost(m_check, temp, offset2, regularized);
			temp.at<double>(i, j) = -eps;
			J -= getCost(m_check, temp, offset2, regularized);
			grad1.at<double>(i, j) = J / 2 / eps;
		}
	}
	//
	for (int i = 0; i < offset2.rows; i++)
	{
		for (int j = 0; j < offset2.cols; j++)
		{
			Mat temp = offset2.clone();
			temp.at<double>(i, j) = eps;
			double J = getCost(m_check, offset1, temp, regularized);
			temp.at<double>(i, j) = -eps;
			J -= getCost(m_check, offset1, temp, regularized);
			grad2.at<double>(i, j) = J / 2 / eps;
		}
	}
	//
	return;
}

double Simple3LayerNN::getCost( NNMats &mats, const Mat &offset1, const Mat &offset2, bool regularized /*= true*/ )
{
	int m = mats.X.rows;
	//
	Mat newt1 = mats.t1 + offset1,
		newt2 = mats.t2 + offset2;
	//
	Mat temp = sigmoid(mats.Xw1 * newt1.t());	
	temp.copyTo(mats.hidW1(Range::all(), Range(1, mats.hidW1.cols)));
	Mat prediction = sigmoid(mats.hidW1 * newt2.t());
	//
	Mat diff = cv::abs(prediction - (1.0 - mats.YSoft));
	Mat logDiff;
	cv::log(diff, logDiff);
	Scalar val = sum(logDiff);
	double J = -val[0] / m;
	if (regularized)
	{
		Mat b1 = newt1.colRange(1, newt1.cols),
			b2 = newt2.colRange(1, newt2.cols);
		Scalar reg = sum(b1.mul(b1)) + sum(b2.mul(b2));
		J += m_params.lambda / (2 * m) * reg[0];
	}
	return J;
}

void Simple3LayerNN::updateWeights( NNMats &mats )
{
	mats.t1 -= m_params.alpha * mats.grad1;
	mats.t2 -= m_params.alpha * mats.grad2;
	m_params.alpha *= 0.99;
}

double Simple3LayerNN::evaluate( NNMats &mats )
{
	Mat temp = sigmoid(mats.Xw1 * mats.t1.t());	
	temp.copyTo(mats.hidW1(Range::all(), Range(1, mats.hidW1.cols)));
	Mat prediction = sigmoid(mats.hidW1 * mats.t2.t());
	int count = 0,
		m = mats.Y.rows,
		n = prediction.cols;
	for(int i = 0; i < m; i++)
	{
		double tmax = -1.0;
		int label = 0;
		for (int j = 0; j < n; j++)
		{
			double pred = prediction.at<double>(i, j);
			if (pred > tmax)
			{
				tmax = pred;
				label = j + 1;
			}
		}
		if (label == (int)mats.Y.at<double>(i, 0))
		{
			count++;
		}
	}
	return (double)count / m;
}