#ifndef CAFFE_CUSTOM_LAYERS_HPP_
#define CAFFE_CUSTOM_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class DeepHandModelDofLimitLossLayer : public LossLayer<Dtype> {
 public:
  explicit DeepHandModelDofLimitLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "DeepHandModelDofLimitLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);	  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);	 
  int C_;
  const float PI = 3.1415926535897932384626;  
  float dofLimitLow[47];
  float dofLimitUp[47];
};


template <typename Dtype>
class DeepHandModelxyzLossLayer : public LossLayer<Dtype> {
public:
  explicit DeepHandModelxyzLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DeepHandModelxyzLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
      
  int model;
  int sum;
  int joint_num[3];
  int dict[3][31];
  vector<int> max_index;
};


  template <typename Dtype>
  class DeepHandModelLayer : public Layer<Dtype> {
   public:
    explicit DeepHandModelLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "DeepHandModel"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top); 
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);    

    
    private:
      class Vec {
      public:
        double V[4];
        Vec(){}
        Vec(double x, double y, double z, double w) {
          V[0] = x;
          V[1] = y;
          V[2] = z;
          V[3] = w;
        }

        inline double operator [](int u) {
          return V[u];
        }

        inline Vec operator +(const Vec &v) {
          return Vec(V[0] + v.V[0], V[1] + v.V[1], V[2] + v.V[2], V[3] + v.V[3]);
        }
      };

      class Matr {
      public:
        double M[4][4];
        Matr(){}
        //only translate 
        Matr(int axis, double x, double y, double z, int opt){
          for (int i = 0; i < 4; i++)
          {
            for (int j = 0; j < 4; j++)
              M[i][j] = 0.0;
          }
          for (int i = 0; i < 4; i++) M[i][i] = (opt == 0) ? 1.0 : 0.0;
          if (opt == 0)
          {
            M[0][3] = x;
            M[1][3] = y;
            M[2][3] = z;
          }
          else if (opt == 1) // gradient w.r.t global offset (x axis)
          {
            M[0][3] = 1.0;
          }
          else if (opt == 2) // gradient w.r.t global offset (y axis)
          {
            M[1][3] = 1.0;
          }
          else               // gradient w.r.t global offset (z axis)
          {
            M[2][3] = 1.0;
          }
        }

        Matr(int axis, double theta, int opt) {     //add a operator to discriminate between forward pass and backward pass  
          for (int i = 0; i < 4; i++)
          for (int j = 0; j < 4; j++)
            M[i][j] = 0;
          if (axis == 0) {
            M[0][0] = (opt == 0) ? 1 : 0;
            M[1][1] = (opt == 0) ? cos(theta) : -sin(theta);
            M[1][2] = (opt == 0) ? -sin(theta) : -cos(theta);
            M[2][1] = (opt == 0) ? sin(theta) : cos(theta);
            M[2][2] = (opt == 0) ? cos(theta) : -sin(theta);
            M[3][3] = (opt == 0) ? 1.0 : 0.0;
          }
          else if (axis == 1) {
            M[0][0] = (opt == 0) ? cos(theta) : -sin(theta);
            M[0][2] = (opt == 0) ? -sin(theta) : -cos(theta);
            M[1][1] = (opt == 0) ? 1 : 0;
            M[2][0] = (opt == 0) ? sin(theta) : cos(theta);
            M[2][2] = (opt == 0) ? cos(theta) : -sin(theta);
            M[3][3] = (opt == 0) ? 1.0 : 0.0;
          }
          else if (axis == 2) {
            M[0][0] = (opt == 0) ? cos(theta) : -sin(theta);
            M[0][1] = (opt == 0) ? -sin(theta) : -cos(theta);
            M[1][0] = (opt == 0) ? sin(theta) : cos(theta);
            M[1][1] = (opt == 0) ? cos(theta) : -sin(theta);
            M[2][2] = (opt == 0) ? 1 : 0;
            M[3][3] = (opt == 0) ? 1.0 : 0.0;
          }

        }

        inline Vec operator *(const Vec &v) {
          return Vec(M[0][0] * v.V[0] + M[0][1] * v.V[1] + M[0][2] * v.V[2] + M[0][3] * v.V[3],
            M[1][0] * v.V[0] + M[1][1] * v.V[1] + M[1][2] * v.V[2] + M[1][3] * v.V[3],
            M[2][0] * v.V[0] + M[2][1] * v.V[1] + M[2][2] * v.V[2] + M[2][3] * v.V[3],
            M[3][0] * v.V[0] + M[3][1] * v.V[1] + M[3][2] * v.V[2] + M[3][3] * v.V[3]);
        }


        inline Matr operator *(const Matr &u) {
          Matr res;
          for (int i = 0; i < 4; i++)
          {
            for (int j = 0; j < 4; j++)
            {
              res.M[i][j] = 0.0;
              for (int k = 0; k < 4; k++) res.M[i][j] += M[i][k] * u.M[k][j];
            }
          }
          return res;
        }
      };
      
      void Update(std::vector<std::pair<int, int> > Rot, int i, const Dtype* bottom_data, int Bid, Vec x);
      Vec f[31][47];
      double initparam[47]; //InitialRotationDegree
      Matr trans[31]; //Initial translation value(decided by bone length)
      Matr rota[23]; //Initial rotation matrices
      double bonelen[30];
  };


}  // namespace caffe

#endif  // CAFFE_COMMON_LAYERS_HPP_
