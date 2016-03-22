#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp> 
#include <opencv2/core/core.hpp>
#include <queue>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <omp.h>
#include <hdf5/hdf5.h>
#include "hdf5/hdf5.h"
#include "hdf5/hdf5_hl.h"
#include <caffe/caffe.hpp>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "HandDefine.h"
#define NumOfModel 10
#define MaxLen 111
#define SIZE 512
#define NumOfConv 3
using namespace std;
using namespace cv;
using namespace caffe;
//Network Configuration
boost::shared_ptr<Net<float> > net;
boost::shared_ptr<Blob<float> > blob;
Mat rgb = Mat::zeros(Size(128, 128), CV_8UC3);
Mat img = Mat::zeros(Size(SIZE, SIZE), CV_8UC3);
int depth_id, pred_joint_id, gt_joint_id, pred_dof_id, gt_dof_id;
int conv_id[NumOfConv];
const float *depth;
const float *pred_joint;
const float *gt_joint;
const float *pred_dof;
const float *gt_dof;
const float *conv_layer[NumOfConv];
const int num_of_feature_map[NumOfConv] = { 8, 8, 8 };
const int feature_map_size[NumOfConv] = { 124, 27, 12 };
char *conv_layer_name[NumOfConv] = { "convL1", "convL2", "convL3" };
char model[NumOfModel][MaxLen];
char proto[NumOfModel][MaxLen];
char *trainornot[2] = { "train", "test" };
void(*EvaluateFunc[NumOfModel])(int opttrainortest, int opt);
char *ExperimentName[NumOfModel] = { "NoBoneRotation", "NoBoneRotation_DoFConstrained", "NoBoneRotation_DP", "NoBoneRotation_JointLearning", "NoBoneRotation_DirectJoint" , "NoBoneRotation_Binary_DirectJoint", "NoBoneRotation_Binary_Model" };
//Display Joint On Image
int edges[30][2] = { { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 4 }, { 4, 20 }, { 5, 6 }, { 6, 7 }, { 7, 8 }, { 8, 9 }, { 9, 21 }, { 10, 11 }, { 11, 12 }, { 12, 13 }, { 13, 14 }, { 14, 22 }, { 15, 16 }, { 16, 17 }, { 17, 18 }, { 18, 19 }, { 19, 23 }, { 20, 24 }, { 21, 24 }, { 22, 24 }, { 23, 24 }, { 24, 25 }, { 24, 26 }, { 24, 27 }, { 27, 28 }, { 28, 29 }, { 29, 30 } };
int NYUJointOrNot[31];
//Calculate error and output to file
int N, BatchSize;
double currenterr[2], err[2];
FILE *foutdof, *foutjoint;
void LoadCNNNetwork(int opt)
{
	Phase phase = TEST;
	Caffe::set_mode(Caffe::GPU);   //if GPU is not used, change mode and uncomment "Caffe::SetDevice(0)"
	//Caffe::set_mode(Caffe::CPU);
	Caffe::SetDevice(0); //change device id of GPU 
	//Load net model	
	net = boost::shared_ptr<Net<float> >(new caffe::Net<float>(proto[opt], phase));
	net->CopyTrainedLayersFrom(model[opt]); //copy model weights from the trained model
}

int FindIdOfBlob(string BlobName)
{
	int i = 0;
	while (net->blob_names()[i]!=BlobName) i++;
	return i;
}

void InitModule(int opttrainortest, int opt)
{
	char filename[111];
	sprintf(filename, "%s%s%s%s%s", "Evaluation/", ExperimentName[opt], "/output/on", trainornot[opttrainortest], "/predictedparameter/parameter.txt");
	foutdof = fopen(filename, "w");
	sprintf(filename, "%s%s%s%s%s", "Evaluation/", ExperimentName[opt], "/output/on", trainornot[opttrainortest], "/predictedjoint/joint.txt");
	foutjoint = fopen(filename, "w");
	err[0] = err[1] = 0.0;
}

void GetBlobValue(int gtdofornot)
{
	//depth
	blob = net->blobs()[depth_id];
	depth = blob->mutable_cpu_data();
	//pred_joint & gt_joint
	blob = net->blobs()[pred_joint_id];
	pred_joint = blob->mutable_cpu_data();
	blob = net->blobs()[gt_joint_id];
	gt_joint = blob->mutable_cpu_data();
	//pred_dof & gt_dof
	blob = net->blobs()[pred_dof_id];
	pred_dof = blob->mutable_cpu_data();
	if (gtdofornot)
	{
		blob = net->blobs()[gt_dof_id];
		gt_dof = blob->mutable_cpu_data();
	}
	for (int i = 0; i < NumOfConv; i++)
	{
		blob = net->blobs()[conv_id[i]];
		conv_layer[i] = blob->mutable_cpu_data();
	}
}

void Display(int opttrainortest, int opt, int i, int bat)
{
	for (int j = 0; j < JointNum; j++)
	{				
		circle(img, Point2d((pred_joint[i * JointNum * 3 + j * 3] + 1.0) * SIZE / 2.0, (-pred_joint[i * JointNum * 3 + j * 3 + 1] + 1.0) * SIZE / 2.0), 3, Scalar(255, 0, 0), -2);
		circle(img, Point2d((gt_joint[i * JointNum * 3 + j * 3] + 1.0) * SIZE / 2.0, (-gt_joint[i * JointNum * 3 + j * 3 + 1] + 1.0) * SIZE / 2.0), 3, Scalar(0, 0, 255), -2);
	}
	
	for (int j = 0; j < BoneNum; j++)
	{
		line(img, Point2d((pred_joint[i * JointNum * 3 + edges[j][0] * 3] + 1.0) * SIZE / 2.0, (-pred_joint[i * JointNum * 3 + edges[j][0] * 3 + 1] + 1.0)*SIZE / 2.0),
			      Point2d((pred_joint[i * JointNum * 3 + edges[j][1] * 3] + 1.0) * SIZE / 2.0, (-pred_joint[i * JointNum * 3 + edges[j][1] * 3 + 1] + 1.0)*SIZE / 2.0), Scalar(255, 0, 0));

		line(img, Point2d((gt_joint[i * JointNum * 3 + edges[j][0] * 3] + 1.0) * SIZE / 2.0, (-gt_joint[i * JointNum * 3 + edges[j][0] * 3 + 1] + 1.0)*SIZE / 2.0),
			      Point2d((gt_joint[i * JointNum * 3 + edges[j][1] * 3] + 1.0) * SIZE / 2.0, (-gt_joint[i * JointNum * 3 + edges[j][1] * 3 + 1] + 1.0)*SIZE / 2.0), Scalar(0, 0, 255));		
	}
	imshow("", img);
	waitKey(1);

	char outputname[111];
	sprintf(outputname, "%s%s%s%s%s%d%s", "Evaluation/", ExperimentName[opt], "/output/on", trainornot[opttrainortest], "/displayjointonfigure/", bat * BatchSize + i, ".png");
	imwrite(outputname, img);

	//output error
	currenterr[0] /= 14.0;
	currenterr[1] /= 31.0;
	err[0] += currenterr[0];
	err[1] += currenterr[1];
	cout << " 14 joint : " << currenterr[0] << " 31 joint : " << currenterr[1] << "\n";
}

void OutputConvFeatureMap(int opttrainortest, int opt, int i, int bat)
{
	for (int l = 0; l < NumOfConv; l++)
	{
		for (int j = 0; j < num_of_feature_map[l]; j++)
		{
			Mat feat = Mat::zeros(Size(feature_map_size[l], feature_map_size[l]), CV_8UC1);			
			for (int h = 0; h < feature_map_size[l]; h++)
			{
				for (int w = 0; w < feature_map_size[l]; w++)
				{					
					feat.at<uchar>(h, w) = (conv_layer[l][i * num_of_feature_map[l] * feature_map_size[l] * feature_map_size[l] + j * feature_map_size[l] * feature_map_size[l] + h * feature_map_size[l] + w] + 1.0) / 2.0  * 255;					
				}
			}
			char outputname[111];
			resize(feat, feat, Size(512, 512));
			sprintf(outputname, "%s%s%s%s%s%s%s%d%s%d%s", "Evaluation/", ExperimentName[opt], "/output/on", trainornot[opttrainortest], "/featuremap/", conv_layer_name[l], "/", j, "/", bat*BatchSize + i,".png");
			imwrite(outputname, feat);
		}
	}	
}

void OutputPredDofJoint(int opttrainortest, int outputdofornot, int i, int bat)
{
	for (int h = 0; h < 128; h++)
	{
		for (int w = 0; w < 128; w++)
		{
			rgb.at<Vec3b>(h, w)[0] = rgb.at<Vec3b>(h, w)[1] = rgb.at<Vec3b>(h, w)[2] = (depth[i * 128 * 128 + h * 128 + w] + 1.0) / 2.0 * 255;
		}
	}
	resize(rgb, img, Size(SIZE, SIZE));
	currenterr[0] = currenterr[1] = 0.0; //Initialization
	double ratio = (opttrainortest == 0 ? 150.0 : (bat * BatchSize + i < 2440 ? 150.0 : 130.0)); //because the size of extracted cube is different between two test people
	cout << "Current Image Id: " << " " << bat * BatchSize + i << " Error: "<<"\n";
	if (outputdofornot)
	{
		for (int j = 0; j < ParamNum; j++)
		{
			fprintf(foutdof, "%15.12f ", pred_dof[i * 47 + j]);
		}
		fprintf(foutdof, "\n");
	}	
	for (int j = 0; j < JointNum; j++)
	{
		fprintf(foutjoint, "%15.12f %15.12f %15.12f ", pred_joint[i * JointNum * 3 + j * 3], pred_joint[i * JointNum * 3 + j * 3 + 1], pred_joint[i * JointNum * 3 + j * 3 + 2]);
		if (NYUJointOrNot[j] == 1) //0: 14 1: 31
		{
			double t = 0.0;
			for (int k = 0; k < 3; k++)
			{
				t += pow(pred_joint[i * JointNum * 3 + j * 3 + k] - gt_joint[i * JointNum * 3 + j * 3 + k], 2);
			}
			currenterr[0] += ratio * sqrt(t);
		}
		double t = 0.0;
		for (int k = 0; k < 3; k++)
		{
			t += pow(pred_joint[i * JointNum * 3 + j * 3 + k] - gt_joint[i * JointNum * 3 + j * 3 + k], 2);
		}
		currenterr[1] += ratio * sqrt(t);

	}
	fprintf(foutjoint, "\n");

}

void EndOfModule()
{
	err[0] /= double(N);
	err[1] /= double(N);
	cout << "Total Error 14 joint : " << err[0] << " 31 joint : " << err[1] << "\n";
	fclose(foutdof);
	fclose(foutjoint);
}

void Evaluate_Ours(int opttrainortest, int opt)
{
	boost::shared_ptr<Blob<float> > blob;
	depth_id = FindIdOfBlob("data");
	pred_joint_id = FindIdOfBlob("DeepHandModelxyz");
	gt_joint_id = FindIdOfBlob("label");
	pred_dof_id = FindIdOfBlob("DoF");
	for (int i = 0; i < NumOfConv; i++) conv_id[i] = FindIdOfBlob(conv_layer_name[i]);
	InitModule(opttrainortest, opt);	
	for (int bat = 0; bat < ceil(double(N) / double(BatchSize)); bat++)
	{
		net->ForwardPrefilled();
		GetBlobValue(0);
		for (int i = 0; i < BatchSize; i++)
		{
			if (bat * BatchSize + i >= N) break;
			OutputPredDofJoint(opttrainortest, 1, i, bat);
			Display(opttrainortest, opt, i, bat);
			OutputConvFeatureMap(opttrainortest, opt, i, bat);
		}
	}
	EndOfModule();
}

void Evaluate_DirectParemterAndJointLearningOfParameterAndJoint(int opttrainortest, int opt)
{		
	depth_id = FindIdOfBlob("depth");
	pred_joint_id = FindIdOfBlob("DeepHandModelxyz");
	gt_joint_id = FindIdOfBlob("jointlabel");
	pred_dof_id = FindIdOfBlob("DoF");
	gt_dof_id = FindIdOfBlob("doflabel");	
	for (int i = 0; i < NumOfConv; i++) conv_id[i] = FindIdOfBlob(conv_layer_name[i]);
	InitModule(opttrainortest, opt);
	for (int bat = 0; bat < ceil(double(N) / double(BatchSize)); bat++)
	{
		net->ForwardPrefilled();
		GetBlobValue(1); //need gt_dof				
		for (int i = 0; i < BatchSize; i++)
		{
			if (bat * BatchSize + i >= N) break;			
			OutputPredDofJoint(opttrainortest, 1, i, bat);
			Display(opttrainortest, opt, i, bat);
			OutputConvFeatureMap(opttrainortest, opt, i, bat);
		}
	}
	EndOfModule();	
}

void Evaluate_DirectJoint(int opttrainortest, int opt)
{
	boost::shared_ptr<Blob<float> > blob;
	depth_id = FindIdOfBlob("data");
	pred_joint_id = FindIdOfBlob("DeepHandModelxyz");
	gt_joint_id = FindIdOfBlob("label");
	for (int i = 0; i < NumOfConv; i++) conv_id[i] = FindIdOfBlob(conv_layer_name[i]);
	InitModule(opttrainortest, opt);
	for (int bat = 0; bat < ceil(double(N) / double(BatchSize)); bat++)
	{
		net->ForwardPrefilled();
		GetBlobValue(0);
		for (int i = 0; i < BatchSize; i++)
		{
			if (bat * BatchSize + i >= N) break;
			OutputPredDofJoint(opttrainortest, 0, i, bat);
			Display(opttrainortest, opt, i, bat);
			OutputConvFeatureMap(opttrainortest, opt, i, bat);
		}
	}
	EndOfModule();
}


void Init()
{
	FILE *finmodelname = fopen("configuration/ModelName.txt", "r");
	int n;
	fscanf(finmodelname, "%d", &n);
	for (int i = 0; i < n; i++) fscanf(finmodelname, "%s", model[i]);
	fclose(finmodelname);
	FILE *finprotoname = fopen("configuration/ProtoName.txt", "r");
	fscanf(finprotoname, "%d", &n);
	for (int i = 0; i < n; i++) fscanf(finprotoname, "%s", proto[i]);
	fclose(finprotoname);
	
	EvaluateFunc[0] = &Evaluate_Ours;
	EvaluateFunc[1] = &Evaluate_Ours;
	EvaluateFunc[2] = &Evaluate_DirectParemterAndJointLearningOfParameterAndJoint;
	EvaluateFunc[3] = &Evaluate_DirectParemterAndJointLearningOfParameterAndJoint;
	EvaluateFunc[4] = &Evaluate_DirectJoint;
	EvaluateFunc[5] = &Evaluate_DirectJoint;
	EvaluateFunc[6] = &Evaluate_Ours;
	int NYUJoint[14] = { 0, 3, 5, 8, 10, 13, 15, 18, 24, 25, 26, 28, 29, 30 };
	for (int i = 0; i < 14; i++) NYUJointOrNot[NYUJoint[i]] = 1; //belongs to NYU Joint
}

int main2()
{
	Init();
	cout << "Choose the one you want to evaluate\n";
	cout << "0: Ours without physical constraint\n";
	cout << "1: Ours with physical constraint\n";
	cout << "2: Direct Parameter\n";
	cout << "3: Joint learning of parameter & joint\n";
	cout << "4: Direct joint\n";		
	cout << "5: Binary image Direct Joint(White: hand ; Black: background)\n";
	cout << "6: Binary image ours\n";

	int opt;
	cin >> opt;
	cout << "Forward on the training set or on the testing set 0 : training set 1: testing set\n";
	int opttrainortest;
	cin >> opttrainortest;
	cout << "Define batchsize: \n";
	cin >> BatchSize;
	N = (opttrainortest == 0 ? 72756 : 8252);
	LoadCNNNetwork(opt);
	EvaluateFunc[opt](opttrainortest,opt);
	return 0;
}