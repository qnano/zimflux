// 2D Gaussian PSF models
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include "DLLMacros.h"
#include "Vector.h"
#include "Estimators/Estimation.h"

class Context;
class Estimator;
class sCMOS_Calibration;

typedef Vector4f Gauss2D_Params;
typedef FisherMatrixType<Gauss2D_Params>::type Gauss2D_FisherMatrix;

// spotList [ x y sigmaX sigmaY intensity ]
CDLL_EXPORT void Gauss2D_Draw(float* image, int imgw, int imgh, float* spotList, int nspots, float addSigma=0.0f);


class ISpotDetectorFactory;

struct Gauss3D_Calibration
{
	Gauss3D_Calibration() {}
	float x[4] = {};
	float y[4] = {};
	float minz=0.0f, maxz=0.0f;
};

CDLL_EXPORT Estimator* Gauss2D_CreatePSF_XYIBg(int roisize, float sigmaX, float sigmaY, bool cuda, Context* ctx=0);
CDLL_EXPORT Estimator* Gauss2D_CreatePSF_XYIBgSigma(int roisize, float initialSigma, bool cuda, Context* ctx=0);
CDLL_EXPORT Estimator* Gauss2D_CreatePSF_XYZIBg(int roisize, const Gauss3D_Calibration& calib, bool cuda, Context* ctx=0);
CDLL_EXPORT Estimator* Gauss2D_CreatePSF_XYIBgSigmaXY(int roisize, float initialSigmaX, float initialSigmaY, bool cuda, Context* ctx);


