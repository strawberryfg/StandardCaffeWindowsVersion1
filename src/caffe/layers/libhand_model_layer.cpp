#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/deep_model_layers.hpp"

namespace caffe {


template <typename Dtype>
void libHandModelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
   FILE *fin=fopen("modelinit.txt","r");
   for (int i = 0; i < 93; i++)
   {
     fscanf(fin,"%f",&initModel[i]);
     initModel[i] *= this->layer_param_.libhand_param().scale();
   }
   fclose(fin);
}


template <typename Dtype>
void libHandModelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = 93;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void libHandModelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {  
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int batSize = (bottom[0]->shape())[0];	
  for (int t = 0; t < batSize; t++) {
    int Bid = t * 55;
    Vec x[31];
    
    //palm center
    Vec opalm(bottom_data[Bid], bottom_data[Bid + 1], bottom_data[Bid + 2]);
    Matr rpalm = Matr(0, bottom_data[Bid + 3]) * Matr(1, bottom_data[Bid + 4]) * Matr(2, bottom_data[Bid + 5]);
    for (int i = 24; i < 25; i++)
      x[i] = rpalm * Vec(initModel[i * 3], initModel[i * 3 + 1], initModel[i * 3 + 2]) + opalm;
    //
    
    //palm root
    Vec opalmcenter = x[24];
    Matr rpalmroot = Matr(0, bottom_data[Bid + 6]) * Matr(1, bottom_data[Bid + 7]) * Matr(2,bottom_data[Bid + 8]);
    for (int i = 25; i < 26; i++)
      x[i] = rpalm * rpalmroot * Vec(initModel[i * 3] - initModel[24 * 3], initModel[i * 3 + 1] - initModel[24 * 3 + 1], initModel[i * 3 + 2] - initModel[24 * 3 + 2]) + opalmcenter;
    rpalmroot = Matr(0, bottom_data[Bid + 9]) * Matr(1, bottom_data[Bid + 10]) * Matr(2,bottom_data[Bid + 11]);
    for (int i = 26; i < 27; i++)
      x[i] = rpalm * rpalmroot * Vec(initModel[i * 3] - initModel[24 * 3], initModel[i * 3 + 1] - initModel[24 * 3 + 1], initModel[i * 3 + 2] - initModel[24 * 3 + 2]) + opalmcenter;
    rpalmroot = Matr(0, bottom_data[Bid + 12]) * Matr(1, bottom_data[Bid + 13]) * Matr(2,bottom_data[Bid + 14]);
    for (int i = 27; i < 28; i++)
      x[i] = rpalm * rpalmroot * Vec(initModel[i * 3] - initModel[24 * 3], initModel[i * 3 + 1] - initModel[24 * 3 + 1], initModel[i * 3 + 2] - initModel[24 * 3 + 2]) + opalmcenter;
    //
    
    //thumb
    Vec othumb = x[27];
    Matr rthumb = Matr(1, bottom_data[Bid + 15]) * Matr(2, bottom_data[Bid + 16]);
    for (int i = 28; i < 29; i++)
      x[i] = rpalm * rpalmroot * rthumb * Vec(initModel[i * 3] - initModel[27 * 3], initModel[i * 3 + 1] - initModel[27 * 3 + 1], initModel[i * 3 + 2] - initModel[27 * 3 + 2]) + othumb;
    Vec othumbsecond = x[28];
    Matr rthumbsecond = Matr(2, bottom_data[Bid + 17]);
    for (int i = 29; i < 30; i++)
      x[i] = rpalm * rpalmroot * rthumb * rthumbsecond * Vec(initModel[i * 3] - initModel[28 * 3], initModel[i * 3 + 1] - initModel[28 * 3 + 1], initModel[i * 3 + 2] - initModel[28 * 3 + 2]) + othumbsecond;
    Vec othumbthird = x[29];
    Matr rthumbthird = Matr(2, bottom_data[Bid + 18]);
    for (int i = 30; i < 31; i++)
      x[i] = rpalm * rpalmroot * rthumb * rthumbsecond * rthumbthird * Vec(initModel[i * 3] - initModel[29 * 3], initModel[i * 3 + 1] - initModel[29 * 3 + 1], initModel[i * 3 + 2] - initModel[29 * 3 + 2]) + othumbthird;
    //

    //Bone 1-4
    opalmcenter = x[24];
    Matr rbone1 = Matr(0, bottom_data[Bid + 19]) * Matr(1, bottom_data[Bid + 20]) * Matr(2, bottom_data[Bid + 21]);
    for (int i = 20; i < 21; i++)
      x[i] = rpalm * rbone1 * Vec(initModel[i * 3] - initModel[24 * 3], initModel[i * 3 + 1] - initModel[24 * 3 + 1], initModel[i * 3 + 2] - initModel[24 * 3 + 2]) + opalmcenter;
    Matr rbone2 = Matr(0, bottom_data[Bid + 22]) * Matr(1, bottom_data[Bid + 23]) * Matr(2, bottom_data[Bid + 24]);
    for (int i = 21; i < 22; i++)
      x[i] = rpalm * rbone2 * Vec(initModel[i * 3] - initModel[24 * 3], initModel[i * 3 + 1] - initModel[24 * 3 + 1], initModel[i * 3 + 2] - initModel[24 * 3 + 2]) + opalmcenter;
    Matr rbone3 = Matr(0, bottom_data[Bid + 25]) * Matr(1, bottom_data[Bid + 26]) * Matr(2, bottom_data[Bid + 27]);
    for (int i = 22; i < 23; i++)
      x[i] = rpalm * rbone3 * Vec(initModel[i * 3] - initModel[24 * 3], initModel[i * 3 + 1] - initModel[24 * 3 + 1], initModel[i * 3 + 2] - initModel[24 * 3 + 2]) + opalmcenter;    
    Matr rbone4 = Matr(0, bottom_data[Bid + 28]) * Matr(1, bottom_data[Bid + 29]) * Matr(2, bottom_data[Bid + 30]);
    for (int i = 23; i < 24; i++)
      x[i] = rpalm * rbone4 * Vec(initModel[i * 3] - initModel[24 * 3], initModel[i * 3 + 1] - initModel[24 * 3 + 1], initModel[i * 3 + 2] - initModel[24 * 3 + 2]) + opalmcenter;        
    //
    
    //Bone 1
    Vec obone1 = x[20];
    Matr rbone1first = Matr(0, bottom_data[Bid + 31]) * Matr(2, bottom_data[Bid + 32]);
    for (int i = 4; i < 5; i++)
      x[i] = rpalm * rbone1 * rbone1first * Vec(initModel[i * 3] - initModel[20 * 3], initModel[i * 3 + 1] - initModel[20 * 3 + 1], initModel[i * 3 + 2] - initModel[20 * 3 + 2]) + obone1;
    Vec obone1second = x[4];
    Matr rbone1second = Matr(0, bottom_data[Bid + 33]) * Matr(2, bottom_data[Bid + 47]);
    for (int i = 2; i < 4; i++)
      x[i] = rpalm * rbone1 * rbone1first * rbone1second * Vec(initModel[i * 3] - initModel[4 * 3], initModel[i * 3 + 1] - initModel[4 * 3 + 1], initModel[i * 3 + 2] - initModel[4 * 3 + 2]) + obone1second;
    Vec obone1third = x[2];
    Matr rbone1third = Matr(0, bottom_data[Bid + 34]) * Matr(2, bottom_data[Bid + 48]);
    for (int i = 0; i < 2; i++)
      x[i] = rpalm * rbone1 * rbone1first * rbone1second * rbone1third * Vec(initModel[i * 3] - initModel[2 * 3], initModel[i * 3 + 1] - initModel[2 * 3 + 1], initModel[i * 3 + 2] - initModel[2 * 3 + 2]) + obone1third;
    //

    //Bone 2
    Vec obone2 = x[21];
    Matr rbone2first = Matr(0, bottom_data[Bid + 35]) * Matr(2, bottom_data[Bid + 36]);
    for (int i = 9; i < 10; i++)
      x[i] = rpalm * rbone2 * rbone2first * Vec(initModel[i * 3] - initModel[21 * 3], initModel[i * 3 + 1] - initModel[21 * 3 + 1], initModel[i * 3 + 2] - initModel[21 * 3 + 2]) + obone2;
    Vec obone2second = x[9];
    Matr rbone2second = Matr(0, bottom_data[Bid + 37]) * Matr(2, bottom_data[Bid + 49]);
    for (int i = 7; i < 9; i++)
      x[i] = rpalm * rbone2 * rbone2first * rbone2second * Vec(initModel[i * 3] - initModel[9 * 3], initModel[i * 3 + 1] - initModel[9 * 3 + 1], initModel[i * 3 + 2] - initModel[9 * 3 + 2]) + obone2second;
    Vec obone2third = x[7];
    Matr rbone2third = Matr(0, bottom_data[Bid + 38]) * Matr(2, bottom_data[Bid + 50]);
    for (int i = 5; i < 7; i++)
      x[i] = rpalm * rbone2 * rbone2first * rbone2second * rbone2third * Vec(initModel[i * 3] - initModel[7 * 3], initModel[i * 3 + 1] - initModel[7 * 3 + 1], initModel[i * 3 + 2] - initModel[7 * 3 + 2]) + obone2third;    
    //

    //Bone 3
    Vec obone3 = x[22];
    Matr rbone3first = Matr(0, bottom_data[Bid + 39]) * Matr(2, bottom_data[Bid + 40]);
    for (int i = 14; i < 15; i++)
      x[i] = rpalm * rbone3 * rbone3first * Vec(initModel[i * 3] - initModel[22 * 3], initModel[i * 3 + 1] - initModel[22 * 3 + 1], initModel[i * 3 + 2] - initModel[22 * 3 + 2]) + obone3;
    Vec obone3second = x[14];
    Matr rbone3second = Matr(0, bottom_data[Bid + 41]) * Matr(2, bottom_data[Bid + 51]);
    for (int i = 12; i < 14; i++)
      x[i] = rpalm * rbone3 * rbone3first * rbone3second * Vec(initModel[i * 3] - initModel[14 * 3], initModel[i * 3 + 1] - initModel[14 * 3 + 1], initModel[i * 3 + 2] - initModel[14 * 3 + 2]) + obone3second;
    Vec obone3third = x[12];
    Matr rbone3third = Matr(0, bottom_data[Bid + 42]) * Matr(2, bottom_data[Bid + 52]);
    for (int i = 10; i < 12; i++)
      x[i] = rpalm * rbone3 * rbone3first * rbone3second * rbone3third * Vec(initModel[i * 3] - initModel[12 * 3], initModel[i * 3 + 1] - initModel[12 * 3 + 1], initModel[i * 3 + 2] - initModel[12 * 3 + 2]) + obone3third;     
    //

    //Bone 4
    Vec obone4 = x[23];
    Matr rbone4first = Matr(0, bottom_data[Bid + 43]) * Matr(2, bottom_data[Bid + 44]);
    for (int i = 19; i < 20; i++)
      x[i] = rpalm * rbone4 * rbone4first * Vec(initModel[i * 3] - initModel[23 * 3], initModel[i * 3 + 1] - initModel[23 * 3 + 1], initModel[i * 3 + 2] - initModel[23 * 3 + 2]) + obone4;
    Vec obone4second = x[19];
    Matr rbone4second = Matr(0, bottom_data[Bid + 45]) * Matr(2, bottom_data[Bid + 53]);
    for (int i = 17; i < 19; i++)
      x[i] = rpalm * rbone4 * rbone4first * rbone4second * Vec(initModel[i * 3] - initModel[19 * 3], initModel[i * 3 + 1] - initModel[19 * 3 + 1], initModel[i * 3 + 2] - initModel[19 * 3 + 2]) + obone4second;
    Vec obone4third = x[17];
    Matr rbone4third = Matr(0, bottom_data[Bid + 46]) * Matr(2, bottom_data[Bid + 54]);
    for (int i = 15; i < 17; i++)
      x[i] = rpalm * rbone4 * rbone4first * rbone4second * rbone4third * Vec(initModel[i * 3] - initModel[17 * 3], initModel[i * 3 + 1] - initModel[17 * 3 + 1], initModel[i * 3 + 2] - initModel[17 * 3 + 2]) + obone4third;         
    //
    int Tid = t * 93;
    for (int i = 0; i < 31; i++) {
      top_data[Tid + i * 3] = x[i][0];
      top_data[Tid + i * 3 + 1] = x[i][1];
      top_data[Tid + i * 3 + 2] = x[i][2];
    }
  }

}


template <typename Dtype>
void libHandModelLayer<Dtype>::update(std::vector<std::pair<int, int> > Rot, int i, const Dtype* bottom_data, int Bid) {
        for (int r = 0; r < Rot.size(); r++) {
          Matr R(Rot[r].first, bottom_data[Bid + Rot[r].second]);
          for (int j = 0; j < 55; j++) {
            float g[3][3];
            for (int k = 0; k < 3; k++) 
              for (int p = 0; p < 3; p++)
                g[k][p] = f[i][j][k][p];

            if (j == Rot[r].second) { 
              if (Rot[r].first == 0) {
                for (int p = 0; p < 3; p++)
                  f[i][j][0][p] = g[0][p];

                f[i][j][1][1] = g[1][0];
                f[i][j][1][2] = -g[2][0];
                
                f[i][j][2][1] = g[2][0];
                f[i][j][2][2] = g[1][0];

                f[i][j][1][0] = 0;
                f[i][j][2][0] = 0;
              } else 
              if (Rot[r].first == 1) {
                for (int p = 0; p < 3; p++)
                  f[i][j][1][p] = g[1][p];

                f[i][j][0][1] = g[0][0];
                f[i][j][0][2] = -g[2][0];
                
                f[i][j][2][1] = g[2][0];
            
                f[i][j][2][2] = g[0][0];

                f[i][j][0][0] = 0;
                f[i][j][2][0] = 0;
              } else 

              if (Rot[r].first == 2) {
                for (int p = 0; p < 3; p ++) 
                  f[i][j][2][p] = g[2][p];
                  
                f[i][j][0][1] = g[0][0];
                f[i][j][0][2] = -g[1][0];

                f[i][j][1][1] = g[1][0];
                f[i][j][1][2] = g[0][0];

                f[i][j][0][0] = 0;
                f[i][j][1][0] = 0;
              }
            } else {
              for (int k = 0; k < 3; k++) 
                for (int p = 0; p < 3; p++) 
                  f[i][j][k][p] = g[0][p] * R.M[k][0] + g[1][p] * R.M[k][1] + g[2][p] * R.M[k][2];
            }
          }
        }  
}

template <typename Dtype>
void libHandModelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int batSize = (bottom[0]->shape())[0];	

    for (int t = 0; t < batSize; t++) {
      int Bid = t * 55;
      for (int i = 0; i < 31; i++)
        for (int j = 0; j < 55; j++)
          for (int k = 0; k < 3; k++)
            for (int p = 0; p < 3; p++)
              f[i][j][k][p] = 0;

      //BP palm center
      std::vector<std::pair<int, int> > Rot;
      Rot.push_back(std::make_pair(2, 5));
      Rot.push_back(std::make_pair(1, 4));
      Rot.push_back(std::make_pair(0, 3));
      for (int i = 24; i < 25; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;    
          }
        }
        update(Rot, i, bottom_data, Bid);
      }
      //
      
      //BP palm root left
      Rot.clear();
      Rot.push_back(std::make_pair(2, 8));
      Rot.push_back(std::make_pair(1, 7));
      Rot.push_back(std::make_pair(0, 6));
      Rot.push_back(std::make_pair(2, 5));
      Rot.push_back(std::make_pair(1, 4));
      Rot.push_back(std::make_pair(0, 3));
      for (int i = 25; i < 26; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[24 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[24][j][k][p];
      }
      //

      //BP palm root right
      Rot.clear();
      Rot.push_back(std::make_pair(2, 11));
      Rot.push_back(std::make_pair(1, 10));
      Rot.push_back(std::make_pair(0, 9));
      Rot.push_back(std::make_pair(2, 5));
      Rot.push_back(std::make_pair(1, 4));
      Rot.push_back(std::make_pair(0, 3));
      for (int i = 26; i < 27; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[24 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[24][j][k][p];
      }
      //
      
      //BP palm root another
      Rot.clear();
      Rot.push_back(std::make_pair(2, 14));
      Rot.push_back(std::make_pair(1, 13));
      Rot.push_back(std::make_pair(0, 12));
      Rot.push_back(std::make_pair(2, 5));
      Rot.push_back(std::make_pair(1, 4));
      Rot.push_back(std::make_pair(0, 3));
      for (int i = 27; i < 28; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[24 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[24][j][k][p];
      }
      //      
     
      //BP thumb
      Rot.clear();
      Rot.push_back(std::make_pair(2, 16));    
      Rot.push_back(std::make_pair(1, 15));
      Rot.push_back(std::make_pair(2, 14));    
      Rot.push_back(std::make_pair(1, 13));
      Rot.push_back(std::make_pair(0, 12));
      Rot.push_back(std::make_pair(2, 5));    
      Rot.push_back(std::make_pair(1, 4));
      Rot.push_back(std::make_pair(0, 3));      
      for (int i = 28; i < 29; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[27 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[27][j][k][p];
      }

      Rot.clear();
      Rot.push_back(std::make_pair(2, 17));          
      Rot.push_back(std::make_pair(2, 16));    
      Rot.push_back(std::make_pair(1, 15));
      Rot.push_back(std::make_pair(2, 14));
      Rot.push_back(std::make_pair(1, 13));
      Rot.push_back(std::make_pair(0, 12));      
      Rot.push_back(std::make_pair(2, 5));    
      Rot.push_back(std::make_pair(1, 4));
      Rot.push_back(std::make_pair(0, 3));      
      for (int i = 29; i < 30; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[28 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[28][j][k][p];
      }

      Rot.clear();
      Rot.push_back(std::make_pair(2, 18));          
      Rot.push_back(std::make_pair(2, 17));          
      Rot.push_back(std::make_pair(2, 16));    
      Rot.push_back(std::make_pair(1, 15));
      Rot.push_back(std::make_pair(2, 14));
      Rot.push_back(std::make_pair(1, 13));
      Rot.push_back(std::make_pair(0, 12));      
      Rot.push_back(std::make_pair(2, 5));    
      Rot.push_back(std::make_pair(1, 4));
      Rot.push_back(std::make_pair(0, 3));           
      for (int i = 30; i < 31; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[29 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[29][j][k][p];
      }
      //

      //BP Bone 1-4
      Rot.clear();
      Rot.push_back(std::make_pair(2, 21));          
      Rot.push_back(std::make_pair(1, 20));          
      Rot.push_back(std::make_pair(0, 19));
      Rot.push_back(std::make_pair(2, 5));
      Rot.push_back(std::make_pair(1, 4));
      Rot.push_back(std::make_pair(0, 3));
      for (int i = 20; i < 21; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[24 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[24][j][k][p];
      }
      //Bone 2
      Rot.clear();
      Rot.push_back(std::make_pair(2, 24));          
      Rot.push_back(std::make_pair(1, 23));          
      Rot.push_back(std::make_pair(0, 22));
      Rot.push_back(std::make_pair(2, 5));
      Rot.push_back(std::make_pair(1, 4));
      Rot.push_back(std::make_pair(0, 3));
      for (int i = 21; i < 22; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[24 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[24][j][k][p];
      }

      Rot.clear();
      Rot.push_back(std::make_pair(2, 27));          
      Rot.push_back(std::make_pair(1, 26));          
      Rot.push_back(std::make_pair(0, 25));
      Rot.push_back(std::make_pair(2, 5));
      Rot.push_back(std::make_pair(1, 4));
      Rot.push_back(std::make_pair(0, 3));
      for (int i = 22; i < 23; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[24 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[24][j][k][p];
      }

      Rot.clear();
      Rot.push_back(std::make_pair(2, 30));          
      Rot.push_back(std::make_pair(1, 29));          
      Rot.push_back(std::make_pair(0, 28));
      Rot.push_back(std::make_pair(2, 5));
      Rot.push_back(std::make_pair(1, 4));
      Rot.push_back(std::make_pair(0, 3));
      for (int i = 23; i < 24; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[24 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[24][j][k][p];
      }      
      //

      //Bone 1
      Rot.clear();
      Rot.push_back(std::make_pair(2, 32));          
      Rot.push_back(std::make_pair(0, 31));          
      Rot.push_back(std::make_pair(2, 21));          
      Rot.push_back(std::make_pair(1, 20));          
      Rot.push_back(std::make_pair(0, 19));          
      Rot.push_back(std::make_pair(2, 5));          
      Rot.push_back(std::make_pair(1, 4));          
      Rot.push_back(std::make_pair(0, 3));          
      for (int i = 4; i < 5; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[20 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[20][j][k][p];
      }      
      //Bone 1 Second       
      Rot.clear();
      Rot.push_back(std::make_pair(2, 47));          
      Rot.push_back(std::make_pair(0, 33));          
      Rot.push_back(std::make_pair(2, 32));          
      Rot.push_back(std::make_pair(0, 31));          
      Rot.push_back(std::make_pair(2, 21));          
      Rot.push_back(std::make_pair(1, 20));          
      Rot.push_back(std::make_pair(0, 19));          
      Rot.push_back(std::make_pair(2, 5));          
      Rot.push_back(std::make_pair(1, 4));          
      Rot.push_back(std::make_pair(0, 3));          
      for (int i = 2; i < 4; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[4 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[4][j][k][p];
      }      
      //Bone 1 Third
      Rot.clear();
      Rot.push_back(std::make_pair(2, 48));          
      Rot.push_back(std::make_pair(0, 34));          
      Rot.push_back(std::make_pair(2, 47));          
      Rot.push_back(std::make_pair(0, 33));          
      Rot.push_back(std::make_pair(2, 32));          
      Rot.push_back(std::make_pair(0, 31));          
      Rot.push_back(std::make_pair(2, 21));          
      Rot.push_back(std::make_pair(1, 20));          
      Rot.push_back(std::make_pair(0, 19));          
      Rot.push_back(std::make_pair(2, 5));          
      Rot.push_back(std::make_pair(1, 4));          
      Rot.push_back(std::make_pair(0, 3));          
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[2 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[2][j][k][p];
      }      
      //

      //Bone 2
      Rot.clear();
      Rot.push_back(std::make_pair(2, 36));          
      Rot.push_back(std::make_pair(0, 35));          
      Rot.push_back(std::make_pair(2, 24));          
      Rot.push_back(std::make_pair(1, 23));          
      Rot.push_back(std::make_pair(0, 22));          
      Rot.push_back(std::make_pair(2, 5));          
      Rot.push_back(std::make_pair(1, 4));          
      Rot.push_back(std::make_pair(0, 3));          
      for (int i = 9; i < 10; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[21 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[21][j][k][p];
      }      
      //Bone 2 Second       
      Rot.clear();
      Rot.push_back(std::make_pair(2, 49));          
      Rot.push_back(std::make_pair(0, 37));          
      Rot.push_back(std::make_pair(2, 36));          
      Rot.push_back(std::make_pair(0, 35));          
      Rot.push_back(std::make_pair(2, 24));          
      Rot.push_back(std::make_pair(1, 23));          
      Rot.push_back(std::make_pair(0, 22));          
      Rot.push_back(std::make_pair(2, 5));          
      Rot.push_back(std::make_pair(1, 4));          
      Rot.push_back(std::make_pair(0, 3));          
      for (int i = 7; i < 9; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[9 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[9][j][k][p];
      }      
      //Bone 2 Third
      Rot.clear();
      Rot.push_back(std::make_pair(2, 50));          
      Rot.push_back(std::make_pair(0, 38));    
      Rot.push_back(std::make_pair(2, 49));                
      Rot.push_back(std::make_pair(0, 37));          
      Rot.push_back(std::make_pair(2, 36));          
      Rot.push_back(std::make_pair(0, 35));          
      Rot.push_back(std::make_pair(2, 24));          
      Rot.push_back(std::make_pair(1, 23));          
      Rot.push_back(std::make_pair(0, 22));          
      Rot.push_back(std::make_pair(2, 5));          
      Rot.push_back(std::make_pair(1, 4));          
      Rot.push_back(std::make_pair(0, 3));                   
      for (int i = 5; i < 7; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[7 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[7][j][k][p];
      }      
      //

      //Bone 3
      Rot.clear();
      Rot.push_back(std::make_pair(2, 40));          
      Rot.push_back(std::make_pair(0, 39));          
      Rot.push_back(std::make_pair(2, 27));          
      Rot.push_back(std::make_pair(1, 26));          
      Rot.push_back(std::make_pair(0, 25));          
      Rot.push_back(std::make_pair(2, 5));          
      Rot.push_back(std::make_pair(1, 4));          
      Rot.push_back(std::make_pair(0, 3));          
      for (int i = 14; i < 15; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[22 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[22][j][k][p];
      }      
      //Bone 3 Second       
      Rot.clear();
      Rot.push_back(std::make_pair(2, 51));          
      Rot.push_back(std::make_pair(0, 41));          
      Rot.push_back(std::make_pair(2, 40));          
      Rot.push_back(std::make_pair(0, 39));          
      Rot.push_back(std::make_pair(2, 27));          
      Rot.push_back(std::make_pair(1, 26));          
      Rot.push_back(std::make_pair(0, 25));          
      Rot.push_back(std::make_pair(2, 5));          
      Rot.push_back(std::make_pair(1, 4));          
      Rot.push_back(std::make_pair(0, 3));               
      for (int i = 12; i < 14; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[14 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[14][j][k][p];
      }      
      //Bone 3 Third
      Rot.clear();
      Rot.push_back(std::make_pair(2, 52));          
      Rot.push_back(std::make_pair(0, 42));          
      Rot.push_back(std::make_pair(2, 51));          
      Rot.push_back(std::make_pair(0, 41));          
      Rot.push_back(std::make_pair(2, 40));          
      Rot.push_back(std::make_pair(0, 39));          
      Rot.push_back(std::make_pair(2, 27));          
      Rot.push_back(std::make_pair(1, 26));          
      Rot.push_back(std::make_pair(0, 25));          
      Rot.push_back(std::make_pair(2, 5));          
      Rot.push_back(std::make_pair(1, 4));          
      Rot.push_back(std::make_pair(0, 3));   
      for (int i = 10; i < 12; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[12 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[12][j][k][p];
      }      
      //

      //Bone 4
      Rot.clear();
      Rot.push_back(std::make_pair(2, 44));          
      Rot.push_back(std::make_pair(0, 43));          
      Rot.push_back(std::make_pair(2, 30));          
      Rot.push_back(std::make_pair(1, 29));          
      Rot.push_back(std::make_pair(0, 28));          
      Rot.push_back(std::make_pair(2, 5));          
      Rot.push_back(std::make_pair(1, 4));          
      Rot.push_back(std::make_pair(0, 3));          
      for (int i = 19; i < 20; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[23 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[23][j][k][p];
      }      
      //Bone 4 Second       
      Rot.clear();
      Rot.push_back(std::make_pair(2, 53));          
      Rot.push_back(std::make_pair(0, 45));          
      Rot.push_back(std::make_pair(2, 44));          
      Rot.push_back(std::make_pair(0, 43));          
      Rot.push_back(std::make_pair(2, 30));          
      Rot.push_back(std::make_pair(1, 29));          
      Rot.push_back(std::make_pair(0, 28));          
      Rot.push_back(std::make_pair(2, 5));          
      Rot.push_back(std::make_pair(1, 4));          
      Rot.push_back(std::make_pair(0, 3));             
      for (int i = 17; i < 19; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[19 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[19][j][k][p];
      }      
      //Bone 4 Third
      Rot.clear();
      Rot.push_back(std::make_pair(2, 54));          
      Rot.push_back(std::make_pair(0, 46));       
      Rot.push_back(std::make_pair(2, 53));             
      Rot.push_back(std::make_pair(0, 45));          
      Rot.push_back(std::make_pair(2, 44));          
      Rot.push_back(std::make_pair(0, 43));          
      Rot.push_back(std::make_pair(2, 30));          
      Rot.push_back(std::make_pair(1, 29));          
      Rot.push_back(std::make_pair(0, 28));          
      Rot.push_back(std::make_pair(2, 5));          
      Rot.push_back(std::make_pair(1, 4));          
      Rot.push_back(std::make_pair(0, 3));        
      for (int i = 15; i < 17; i++) {
        for (int j = 0; j < 55; j++) {
          for (int k = 0; k < 3; k++) {
            f[i][j][k][0] = initModel[i * 3 + k] - initModel[17 * 3 + k];
            f[i][j][k][1] = 0;
            f[i][j][k][2] = 0;
          }
        }
        update(Rot, i, bottom_data, Bid);
        for (int j = 0; j < 55; j++)
         for (int k = 0; k < 3; k++)
           for (int p = 0; p < 3; p++)
             f[i][j][k][p] += f[17][j][k][p];
      }      
      //

      for (int j = 0; j < 55; j++) {
        bottom_diff[Bid + j] = 0;
        for (int i = 0; i < 31; i++) {
          int Tid = t * 93 + i * 3;
          if (j < 3) 
              bottom_diff[Bid + j] += top_diff[Tid + j];
          else {
            bottom_diff[Bid + j] += top_diff[Tid] * (f[i][j][0][1] * -sin(bottom_data[Bid + j]) + f[i][j][0][2] * cos(bottom_data[Bid + j]));
            bottom_diff[Bid + j] += top_diff[Tid + 1] * (f[i][j][1][1] * -sin(bottom_data[Bid + j]) + f[i][j][1][2] * cos(bottom_data[Bid + j]));
            bottom_diff[Bid + j] += top_diff[Tid + 2] * (f[i][j][2][1] * -sin(bottom_data[Bid + j]) + f[i][j][2][2] * cos(bottom_data[Bid + j])); 
          }
        }
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(libHandModelLayer);
#endif

INSTANTIATE_CLASS(libHandModelLayer);
REGISTER_LAYER_CLASS(libHandModel);
}  // namespace caffe
