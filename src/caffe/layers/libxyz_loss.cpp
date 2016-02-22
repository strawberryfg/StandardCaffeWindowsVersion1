#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_model_layers.hpp"

namespace caffe {

template <typename Dtype>
void libxyzLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }
  
  if (this->layer_param_.libxyz_loss_param().model() == "LIBHAND") 
    model = 0;
  else 
    if (this->layer_param_.libxyz_loss_param().model() == "NYU") 
      model = 1;
    else 
      if (this->layer_param_.libxyz_loss_param().model() == "ICVL")
        model = 2;
      else {
        std::cerr << "Model Not Find!" << std::endl; 
        exit(1);
      }
}


template <typename Dtype>
void libxyzLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);
  top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void libxyzLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int batSize = (bottom[0]->shape())[0];
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label_data= bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int t = 0; t < batSize; t++) {
    int Biddata = t * 93;
    int Bidlabel = (model < 2? t * 93: t * 48);

    for (int i = 0; i < joint_num[model]; i++) {
      if (model == 1) 
        loss += (bottom_data[Biddata + dict[model][i] * 3] - label_data[Bidlabel + dict[model][i] * 3]) * (bottom_data[Biddata + dict[model][i] * 3] - label_data[Bidlabel + dict[model][i] * 3])+ 
            (bottom_data[Biddata + dict[model][i] * 3 + 1] - label_data[Bidlabel + dict[model][i] * 3 + 1]) * (bottom_data[Biddata + dict[model][i] * 3 + 1] - label_data[Bidlabel + dict[model][i] * 3 + 1])+ 
            (bottom_data[Biddata + dict[model][i] * 3 + 2] - label_data[Bidlabel + dict[model][i] * 3 + 2]) * (bottom_data[Biddata + dict[model][i] * 3 + 2] - label_data[Bidlabel + dict[model][i] * 3 + 2]);
      else
        loss += (bottom_data[Biddata + dict[model][i] * 3] - label_data[Bidlabel + i * 3]) * (bottom_data[Biddata +  dict[model][i] * 3] - label_data[Bidlabel + i * 3]) + 
            (bottom_data[Biddata + dict[model][i] * 3 + 1] - label_data[Bidlabel + i * 3 + 1]) * (bottom_data[Biddata +  dict[model][i] * 3 + 1] - label_data[Bidlabel + i * 3 + 1]) + 
            (bottom_data[Biddata + dict[model][i] * 3 + 2] - label_data[Bidlabel + i * 3 + 2]) * (bottom_data[Biddata +  dict[model][i] * 3 + 2] - label_data[Bidlabel + i * 3 + 2]);
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / batSize / joint_num[model];
}


template <typename Dtype>
void libxyzLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int batSize = (bottom[0]->shape())[0];
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  float top_diff = top[0]->cpu_diff()[0] / batSize / joint_num[model];
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  
  if (propagate_down[0]) {
    for (int t = 0; t < batSize; t++) {
      int Biddata = t * 93;
      int Bidlabel = (model < 2? t * 93: t * 48);
      for (int i = 0; i < 93; i++) 
        bottom_diff[Biddata + i] = 0;

      for (int i = 0; i < joint_num[model]; i++) {
        if (model == 1) {
          bottom_diff[Biddata + dict[model][i] * 3] = top_diff * 2 * (bottom_data[Biddata + dict[model][i] * 3] - label_data[Bidlabel + dict[model][i] * 3]);
          bottom_diff[Biddata + dict[model][i] * 3 + 1] = top_diff * 2 * (bottom_data[Biddata + dict[model][i] * 3 + 1] - label_data[Bidlabel + dict[model][i] * 3 + 1]);
          bottom_diff[Biddata + dict[model][i] * 3 + 2] = top_diff * 2 * (bottom_data[Biddata + dict[model][i] * 3 + 2] - label_data[Bidlabel + dict[model][i] * 3 + 2]);
        } else {
          bottom_diff[Biddata + dict[model][i] * 3] = top_diff * 2 * (bottom_data[Biddata + dict[model][i] * 3] - label_data[Bidlabel + i * 3]);
          bottom_diff[Biddata + dict[model][i] * 3 + 1] = top_diff * 2 * (bottom_data[Biddata + dict[model][i] * 3 + 1] - label_data[Bidlabel + i * 3 + 1]);
          bottom_diff[Biddata + dict[model][i] * 3 + 2] = top_diff * 2 * (bottom_data[Biddata + dict[model][i] * 3 + 2] - label_data[Bidlabel + i * 3 + 2]);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(libxyzLossLayer);
#endif

INSTANTIATE_CLASS(libxyzLossLayer);
REGISTER_LAYER_CLASS(libxyzLoss);

}  // namespace caffe
