#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/custom_layers.hpp"

namespace caffe {

template <typename Dtype>
void DeepHandModelxyzLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }
}


template <typename Dtype>
void DeepHandModelxyzLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);
  top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void DeepHandModelxyzLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int batSize = (bottom[0]->shape())[0];
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label_data= bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int t = 0; t < batSize; t++) {
    int Biddata = t * 93, Bidlabel = t * 93;
    for (int i = 0; i < 31; i++) {
      int coff=0;
      if (i==0 || i==3 || i==5 || i==8 || i==10 || i==13 || i==15 || i==18 || i==24 || i==25 || i==26 || i==28 || i==29 || i==30)
      {
          coff=1;
      }
      else 
      {
          coff=1;
      }
      loss += coff*(bottom_data[Biddata + i * 3] - label_data[Bidlabel + i * 3]) * (bottom_data[Biddata + i * 3] - label_data[Bidlabel + i * 3]) + 
              coff*(bottom_data[Biddata + i * 3 + 1] - label_data[Bidlabel + i * 3 + 1]) * (bottom_data[Biddata + i * 3 + 1] - label_data[Bidlabel + i * 3 + 1]) +
              coff*(bottom_data[Biddata + i * 3 + 2] - label_data[Bidlabel + i * 3 + 2]) * (bottom_data[Biddata + i * 3 + 2] - label_data[Bidlabel + i * 3 + 2]);
    }      
  }
  top[0]->mutable_cpu_data()[0] = loss / batSize /31.0;
}


template <typename Dtype>
void DeepHandModelxyzLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int batSize = (bottom[0]->shape())[0];
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  float top_diff = top[0]->cpu_diff()[0] / batSize /31.0;
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  
  if (propagate_down[0]) {
    for (int t = 0; t < batSize; t++) {
      int Biddata = t * 93, Bidlabel = t * 93;
      for (int i = 0; i < 93; i++) bottom_diff[Biddata + i]=0;
      for (int i = 0; i < 31; i++) {
        int coff=0;
        if (i==0 || i==3 || i==5 || i==8 || i==10 || i==13 || i==15 || i==18 || i==24 || i==25 || i==26 || i==28 || i==29 || i==30)
        {
            coff=1;
        }
        else 
        {
            coff=1;
        }
        bottom_diff[Biddata + i * 3] = top_diff * 2 * coff * (bottom_data[Biddata + i * 3] - label_data[Bidlabel + i * 3]);
        bottom_diff[Biddata + i * 3 + 1] = top_diff * 2 * coff * (bottom_data[Biddata + i * 3 + 1] - label_data[Bidlabel + i * 3 + 1]);
        bottom_diff[Biddata + i * 3 + 2] = top_diff * 2 * coff * (bottom_data[Biddata + i * 3 + 2] - label_data[Bidlabel + i * 3 + 2]);
      }      
    }
  }
  /*
  bottom_diff = bottom[1]->mutable_cpu_diff();

  if (propagate_down[1]) {
    //std::cout<<1<<"\n";
    for (int t = 0; t < batSize; t++) {
      int Biddata = t * 93, Bidlabel = t * 93;
      for (int i = 0; i < 93; i++) bottom_diff[Biddata + i]=0;
      for (int i = 0; i < 31; i++) {
        int coff=0;
        if (i==0 || i==3 || i==5 || i==8 || i==10 || i==13 || i==15 || i==18 || i==24 || i==25 || i==26 || i==28 || i==29 || i==30)
        {
            coff=10;
        }
        else 
        {
            coff=1;
        }        
        bottom_diff[Biddata + i * 3] = top_diff * -2 * (bottom_data[Biddata + i * 3] - label_data[Bidlabel + i * 3]);
        bottom_diff[Biddata + i * 3 + 1] = top_diff * -2 * (bottom_data[Biddata + i * 3 + 1] - label_data[Bidlabel + i * 3 + 1]);
        bottom_diff[Biddata + i * 3 + 2] = top_diff * -2 * (bottom_data[Biddata + i * 3 + 2] - label_data[Bidlabel + i * 3 + 2]);
      }      
    }
  }
  */
}



#ifdef CPU_ONLY
STUB_GPU(DeepHandModelxyzLossLayer);
#endif

INSTANTIATE_CLASS(DeepHandModelxyzLossLayer);
REGISTER_LAYER_CLASS(DeepHandModelxyzLoss);

}  // namespace caffe
