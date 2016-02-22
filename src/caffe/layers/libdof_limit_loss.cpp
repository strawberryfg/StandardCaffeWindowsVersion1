#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_model_layers.hpp"

namespace caffe {

template <typename Dtype>
void libDofLimitLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  FILE *fin = fopen("constraint.txt", "r");
	for (int i = 0; i < 55; i++) 
	{
    fscanf(fin, "%f", &dofLimitLow[i]);
    fscanf(fin, "%f", &dofLimitUp[i]);
	}
  fclose(fin);
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }
  C_ = 1;
}


template <typename Dtype>
void libDofLimitLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);
  top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void libDofLimitLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int batSize = (bottom[0]->shape())[0];
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype loss = 0;
  for (int t = 0; t < batSize; t++) {
    int Bid = t * 55;
    for (int i = 0; i < 55; i++) 
      if (bottom_data[Bid + i] > dofLimitUp[i])
        loss += (bottom_data[Bid + i] - dofLimitUp[i]) * (bottom_data[Bid + i] - dofLimitUp[i]);
      else 
        if (bottom_data[Bid + i] < dofLimitLow[i])
          loss += (bottom_data[Bid + i] - dofLimitLow[i]) * (bottom_data[Bid + i] - dofLimitLow[i]);
  }
  top[0]->mutable_cpu_data()[0] = C_ * loss / batSize;
}

template <typename Dtype>
void libDofLimitLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int batSize = (bottom[0]->shape())[0];
  const Dtype* bottom_data = bottom[0]->cpu_data();
  float top_diff = top[0]->cpu_diff()[0] / batSize;
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  
  if (propagate_down[0]) {
    for (int t = 0; t < batSize; t++) {
      int Bid = t * 55;
      for (int i = 0; i < 55; i++)
        if (bottom_data[Bid + i] > dofLimitUp[i])
          bottom_diff[Bid + i] = top_diff * C_ * 2 * (bottom_data[Bid + i] - dofLimitUp[i]);
        else 
          if (bottom_data[Bid + i] < dofLimitLow[i])
            bottom_diff[Bid + i] = top_diff * C_ * 2 * (bottom_data[Bid + i] - dofLimitLow[i]);
          else 
            bottom_diff[Bid + i] = 0;
    }
  }
}


INSTANTIATE_CLASS(libDofLimitLossLayer);
REGISTER_LAYER_CLASS(libDofLimitLoss);

}  // namespace caffe
