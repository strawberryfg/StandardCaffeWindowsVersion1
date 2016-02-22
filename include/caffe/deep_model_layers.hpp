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
class ScaleLayer : public Layer<Dtype> {
 public:
  explicit ScaleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Scale"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

};


template <typename Dtype>
class libxyzLossLayer : public LossLayer<Dtype> {
public:
  explicit libxyzLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "libxyzLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
      
  int model;
  const int joint_num[3] = {31, 14, 16};
  const int dict[3][31] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30},
    {0, 3, 5, 8, 10, 13, 15, 18, 24, 25, 26, 28, 29, 30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {24, 28, 29, 30, 19, 17, 15, 14, 12, 10, 9, 7, 5, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
  };
};

template <typename Dtype>
class libDofLimitLossLayer : public LossLayer<Dtype> {
 public:
  explicit libDofLimitLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "libDofLimitLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  int C_;
  const float PI = 3.1415926535897932384626;  
  const float dofLimitLow[55] = {-2.0,-2.0,-1.0,-PI*65/180,-PI,-PI,-PI*60/180,-PI*60/180,-PI*60/180,-PI*60/180,-PI*60/180,-PI*60/180,-PI*15/180,-PI*40/180,-PI*60/180,-PI*40/180,-PI*60/180,-PI*30/180,-PI*30/180,-PI*60/180,-PI*40/180,-PI*20/180,-PI*60/180,-PI*40/180,-PI*20/180,-PI*60/180,-PI*40/180,-PI*20/180,-PI*60/180,-PI*40/180,-PI*20/180,-PI*50/180,-PI*20/180,0,0,-PI*50/180,-PI*20/180,0,0,-PI*50/180,-PI*20/180,0,0,-PI*50/180,-PI*20/180,0,0,-PI*20/180,-PI*20/180,-PI*20/180,-PI*20/180,-PI*20/180,-PI*20/180,-PI*20/180,-PI*20/180};                            
  const float dofLimitUp[55] = {
                     2.0,2.0,1.0,                          
                     PI,PI,PI,                             
                     PI*60/180,PI*60/180,PI*60/180,        
                     PI*60/180,PI*60/180,PI*60/180,        
                     PI*30/180,PI*40/180,PI*60/180,        
                     PI*40/180,PI*25/180,                  
                     PI*90/180,                            
                     PI*90/180,                               
                     PI*40/180,PI*60/180,PI*20/180,           
                     PI*40/180,PI*60/180,PI*20/180,           
                     PI*40/180,PI*60/180,PI*20/180,           
                     PI*40/180,PI*60/180,PI*20/180,           
                     PI*120/180,PI*20/180,                    
                     PI*90/180,                               
                     PI*90/180,                               
                     PI*120/180,PI*20/180,                    
                     PI*90/180,                               
                     PI*90/180,                               
                     PI*120/180,PI*20/180,                    
                     PI*90/180,                               
                     PI*90/180,                               
                     PI*120/180,PI*20/180,                    
                     PI*90/180, 
                     PI*90/180, 
                     PI*20/180, 
                     PI*20/180, 
                     PI*20/180, 
                     PI*20/180, 
                     PI*20/180, 
                     PI*20/180, 
                     PI*20/180, 
                     PI*20/180};
 };


template <typename Dtype>
class libHandModelLayer : public Layer<Dtype> {
 public:
  explicit libHandModelLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "libHandModel"; }
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
      float V[3];
      Vec(){}
      Vec(float x, float y, float z) {
        V[0] = x;
        V[1] = y;
        V[2] = z;
      }

      inline float operator [](int u) {
        return V[u];
      }      

      inline Vec operator +(const Vec &v) {
        return Vec(V[0]+v.V[0],V[1]+v.V[1],V[2]+v.V[2]);
      }

      inline Vec operator /(const float &v){
        return Vec(V[0]/v,V[1]/v,V[2]/v);
      }
    };

    class Matr {
    public:
      float M[3][3];
      Matr(){}
      Matr(int axis, float theta) {
        for (int i = 0; i < 3; i++) 
          for (int j = 0; j < 3; j++)
            M[i][j] = 0;
        if (axis == 0) {
          M[0][0] = 1;
          M[1][1] = cos(theta);
          M[1][2] = -sin(theta);
          M[2][1] = sin(theta);
          M[2][2] = cos(theta);
        } else 
        if (axis == 1) {
          M[0][0] = cos(theta);
          M[0][2] = -sin(theta);
          M[1][1] = 1;
          M[2][0] = sin(theta);
          M[2][2] = cos(theta);
        } else 
        if (axis == 2) {
          M[0][0] = cos(theta);
          M[0][1] = -sin(theta);
          M[1][0] = sin(theta);
          M[1][1] = cos(theta);
          M[2][2] = 1;
        }
      }

      inline Vec operator *(const Vec &v) {
        return Vec(M[0][0] * v.V[0] + M[0][1] * v.V[1] + M[0][2] * v.V[2],
                   M[1][0] * v.V[0] + M[1][1] * v.V[1] + M[1][2] * v.V[2],
                   M[2][0] * v.V[0] + M[2][1] * v.V[1] + M[2][2] * v.V[2]);
      }


      inline Matr operator *(const Matr &u) {
        Matr res;
        res.M[0][0] = M[0][0] * u.M[0][0] + M[0][1] * u.M[1][0] + M[0][2] * u.M[2][0];
        res.M[0][1] = M[0][0] * u.M[0][1] + M[0][1] * u.M[1][1] + M[0][2] * u.M[2][1];
        res.M[0][2] = M[0][0] * u.M[0][2] + M[0][1] * u.M[1][2] + M[0][2] * u.M[2][2];

        res.M[1][0] = M[1][0] * u.M[0][0] + M[1][1] * u.M[1][0] + M[1][2] * u.M[2][0];
        res.M[1][1] = M[1][0] * u.M[0][1] + M[1][1] * u.M[1][1] + M[1][2] * u.M[2][1];
        res.M[1][2] = M[1][0] * u.M[0][2] + M[1][1] * u.M[1][2] + M[1][2] * u.M[2][2];

        res.M[2][0] = M[2][0] * u.M[0][0] + M[2][1] * u.M[1][0] + M[2][2] * u.M[2][0];
        res.M[2][1] = M[2][0] * u.M[0][1] + M[2][1] * u.M[1][1] + M[2][2] * u.M[2][1];
        res.M[2][2] = M[2][0] * u.M[0][2] + M[2][1] * u.M[1][2] + M[2][2] * u.M[2][2];
        return res;
      }
    };

    void update(std::vector<std::pair<int, int> > Rot, int i, const Dtype* bottom_data, int Bid);

    float f[31][55][3][3];
    float initModel[93];

};


}  // namespace caffe

#endif  // CAFFE_COMMON_LAYERS_HPP_
