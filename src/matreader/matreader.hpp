#ifndef _MAT_READER_HPP
#define _MAT_READER_HPP

#include "mat.h"

#include "MxArray.h"

#include <opencv2/core/core.hpp>

#include <iostream>

using namespace cv;

class MatFileReader
{
public:
	MatFileReader() { m_matFILE = NULL; };

	MatFileReader(const char *filename) 
	{
		m_matFILE = NULL;
		loadMatFile(filename);
	};

	~MatFileReader()
	{
		fileClose();
	}

	bool loadMatFile(const char *filename)
	{
		fileClose();
		m_matFILE = matOpen(filename, "r");
		if (m_matFILE == NULL)
		{
			std::cout << "Can not open mat file " << filename << " !\n";
			return false;
		}
		return true;
	}

	Mat getVariable(const char * name, int type = 0, bool transpose = true)
	{
		if (m_matFILE == NULL)
		{				
			return Mat();
		}
		mxArray *temp = matGetVariable(m_matFILE, name);
		if (temp == NULL)
		{
			std::cout << "Can not get variable " << name << " !\n";
			return Mat();
		}
		if (type == 0)
		{
			return MxArray(temp).toMatND(CV_USRTYPE1, transpose);
		}
		else
			return MxArray(temp).toMat(CV_USRTYPE1, transpose);
	}

private:
	MATFile *m_matFILE;

protected:
	void fileClose()
	{
		if (m_matFILE != NULL)
		{
			matClose(m_matFILE);
		}
	}

};

#endif	// matreader.hpp