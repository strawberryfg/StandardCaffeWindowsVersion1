#include <algorithm>


#include "caffe/layer.hpp"
#include "caffe/custom_layers.hpp"

namespace caffe {


template <typename Dtype>
void libHandModelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	FILE *fin = fopen("configuration/InitialParameters.in", "r");
	for (int i = 0; i < 47; i++)
	{
		fscanf(fin, "%lf", &initparam[i]);
	}
	fclose(fin);   
	fin = fopen("configuration/BoneLength.txt", "r");
	for (int i = 0; i < 30; i++)
	{
		fscanf(fin, "%lf", &bonelen[i]);
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
    int Bid = t * 47;
    Vec x[31];
    
	Matr rotpalm = Matr(0, bottom_data[Bid + 3], 0) * Matr(1, bottom_data[Bid + 4], 0) * Matr(2, bottom_data[Bid + 5], 0);
	Matr transpalm = Matr(3, bottom_data[Bid], bottom_data[Bid + 1], bottom_data[Bid + 2], 0);
	Matr rotpalminit = Matr(0, initparam[3], 0) * Matr(1, initparam[4], 0) * Matr(2, initparam[5], 0) * Matr(3, initparam[0], initparam[1], initparam[2], 0);
	rota[0] = rotpalminit;
	for (int i = 24; i < 25; i++)
		x[i] = rotpalm * transpalm * rotpalminit * Vec(0.0, 0.0, 0.0, 1.0);
	//

	//palm root

	Matr rotpalmrootleft = Matr(0, bottom_data[Bid + 6], 0) * Matr(1, bottom_data[Bid + 7], 0) * Matr(2, bottom_data[Bid + 8], 0);
	Matr transpalmrootleft = Matr(3, 0.0, -bonelen[24], 0.0, 0);
	Matr rotpalmrootleftinit = Matr(0, initparam[6], 0) * Matr(1, initparam[7], 0) * Matr(2, initparam[8], 0);
	trans[0] = transpalmrootleft;
	rota[1] = rotpalmrootleftinit;
	for (int i = 25; i < 26; i++)
		x[i] = rotpalm * transpalm * rotpalminit * rotpalmrootleft * rotpalmrootleftinit * transpalmrootleft * Vec(0.0, 0.0, 0.0, 1.0);
	
	Matr rotpalmrootmiddle = Matr(0, bottom_data[Bid + 9], 0) * Matr(1, bottom_data[Bid + 10], 0) * Matr(2, bottom_data[Bid + 11], 0);
	Matr transpalmrootmiddle = Matr(3, 0.0, -bonelen[25], 0.0, 0);
	Matr rotpalmrootmiddleinit = Matr(0, initparam[9], 0) * Matr(1, initparam[10], 0) * Matr(2, initparam[11], 0);
	trans[1] = transpalmrootmiddle;
	rota[2] = rotpalmrootmiddleinit;
	for (int i = 26; i < 27; i++)
		x[i] = rotpalm * transpalm * rotpalminit * rotpalmrootmiddle * rotpalmrootmiddleinit * transpalmrootmiddle * Vec(0.0, 0.0, 0.0, 1.0);

	Matr rotpalmrootright = Matr(0, bottom_data[Bid + 12], 0) * Matr(1, bottom_data[Bid + 13], 0) * Matr(2, bottom_data[Bid + 14], 0);
	Matr transpalmrootright = Matr(3, 0.0, -bonelen[26], 0.0, 0);
	Matr rotpalmrootrightinit = Matr(0, initparam[12], 0) * Matr(1, initparam[13], 0) * Matr(2, initparam[14], 0);
	trans[2] = transpalmrootright;
	rota[3] = rotpalmrootrightinit;
	for (int i = 27; i < 28; i++)
		x[i] = rotpalm * transpalm * rotpalminit * rotpalmrootright * rotpalmrootrightinit * transpalmrootright * Vec(0.0, 0.0, 0.0, 1.0);
	//

	//thumb
	Matr rotthumb = Matr(1, bottom_data[Bid + 15], 0) * Matr(2, bottom_data[Bid + 16], 0);
	Matr transthumb = Matr(3, bonelen[27], 0.0, 0.0, 0);
	Matr rotthumbinit = Matr(1, initparam[15], 0) * Matr(2, initparam[16], 0);
	trans[3] = transthumb;
	rota[4] = rotthumbinit;
	for (int i = 28; i < 29; i++)
		x[i] = rotpalm * transpalm * rotpalminit * rotpalmrootright * rotpalmrootrightinit * transpalmrootright * rotthumb * rotthumbinit * transthumb * Vec(0.0, 0.0, 0.0, 1.0);

	Matr rotthumbsecond = Matr(2, bottom_data[Bid + 17], 0);
	Matr transthumbsecond = Matr(3, bonelen[28], 0.0, 0.0, 0);
	Matr rotthumbsecondinit = Matr(2, initparam[17], 0);
	trans[4] = transthumbsecond;
	rota[5] = rotthumbsecondinit;
	for (int i = 29; i < 30; i++)
		x[i] = rotpalm * transpalm * rotpalminit * rotpalmrootright * rotpalmrootrightinit * transpalmrootright * rotthumb * rotthumbinit * transthumb * rotthumbsecond * rotthumbsecondinit * transthumbsecond * Vec(0.0, 0.0, 0.0, 1.0);

	Matr rotthumbthird = Matr(2, bottom_data[Bid + 18], 0);
	Matr transthumbthird = Matr(3, bonelen[29], 0.0, 0.0, 0);
	Matr rotthumbthirdinit = Matr(2, initparam[18], 0);
	trans[5] = transthumbthird;
	rota[6] = rotthumbthirdinit;
	for (int i = 30; i < 31; i++)
		x[i] = rotpalm * transpalm * rotpalminit * rotpalmrootright * rotpalmrootrightinit * transpalmrootright * rotthumb * rotthumbinit * transthumb * rotthumbsecond * rotthumbsecondinit * transthumbsecond * rotthumbthird * rotthumbthirdinit * transthumbthird * Vec(0.0, 0.0, 0.0, 1.0);
	//

	//Finger 1-4
	for (int k = 0; k < 4; k++)
	{
		Matr rotbone = Matr(0, bottom_data[Bid + 19 + 3 * k], 0) * Matr(1, bottom_data[Bid + 20 + 3 * k], 0) * Matr(2, bottom_data[Bid + 21 + 3 * k], 0);
		Matr transbone = Matr(3, 0.0, bonelen[20 + k], 0.0, 0);
		Matr rotboneinit = Matr(0, initparam[19 + 3 * k], 0) * Matr(1, initparam[20 + 3 * k], 0) * Matr(2, initparam[21 + 3 * k], 0);
		trans[6 + k] = transbone;
		rota[7 + k] = rotboneinit;
		for (int i = 20 + k; i < 21 + k; i++)
			x[i] = rotpalm * transpalm * rotpalminit * rotbone * rotboneinit * transbone * Vec(0.0, 0.0, 0.0, 1.0);

		//detailed
		Matr rotfingerfirst = Matr(0, bottom_data[Bid + 31 + 4 * k], 0) * Matr(2, bottom_data[Bid + 32 + 4 * k], 0);
		Matr transfingerfirst = Matr(3, 0.0, bonelen[4 + 5 * k], 0.0, 0);
		Matr rotfingerfirstinit = Matr(0, initparam[31 + 4 * k], 0) * Matr(2, initparam[32 + 4 * k], 0);
		trans[10 + 5 * k] = transfingerfirst;
		rota[11 + 3 * k] = rotfingerfirstinit;
		for (int i = 4 + 5 * k; i > 3 + 5 * k; i--)
			x[i] = rotpalm * transpalm * rotpalminit * rotbone * rotboneinit * transbone * rotfingerfirst * rotfingerfirstinit * transfingerfirst * Vec(0.0, 0.0, 0.0, 1.0);

		Matr rotfingersecond = Matr(0, bottom_data[Bid + 33 + 4 * k], 0);
		Matr transfingersecond = Matr(3, 0.0, bonelen[3 + 5 * k], 0.0, 0);
		Matr rotfingersecondinit = Matr(0, initparam[33 + 4 * k], 0);
		trans[11 + 5 * k] = Matr(3, 0.0, bonelen[3 + 5 * k], 0.0, 0);
		trans[12 + 5 * k] = Matr(3, 0.0, bonelen[2 + 5 * k], 0.0, 0);
		rota[12 + 3 * k] = rotfingersecondinit;
		for (int i = 3 + 5 * k; i > 1 + 5 * k; i--)
		{
			if (i != 3 + 5 * k) transfingersecond = transfingersecond * Matr(3, 0.0, bonelen[2 + 5 * k], 0.0, 0);
			x[i] = rotpalm * transpalm * rotpalminit * rotbone * rotboneinit * transbone * rotfingerfirst * rotfingerfirstinit * transfingerfirst * rotfingersecond * rotfingersecondinit * transfingersecond * Vec(0.0, 0.0, 0.0, 1.0);
		}

		Matr rotfingerthird = Matr(0, bottom_data[Bid + 34 + 4 * k], 0);
		Matr transfingerthird = Matr(3, 0.0, bonelen[1 + 5 * k], 0.0, 0);
		Matr rotfingerthirdinit = Matr(0, initparam[34 + 4 * k], 0);
		trans[13 + 5 * k] = Matr(3, 0.0, bonelen[1 + 5 * k], 0.0, 0);
		trans[14 + 5 * k] = Matr(3, 0.0, bonelen[0 + 5 * k], 0.0, 0);
		rota[13 + 3 * k] = rotfingerthirdinit;
		for (int i = 1 + 5 * k; i > -1 + 5 * k; i--)
		{
			if (i != 1 + 5 * k) transfingerthird = transfingerthird*Matr(3, 0.0, bonelen[0 + 5 * k], 0.0, 0);
			x[i] = rotpalm * transpalm * rotpalminit * rotbone * rotboneinit * transbone * rotfingerfirst * rotfingerfirstinit * transfingerfirst * rotfingersecond * rotfingersecondinit * transfingersecond * rotfingerthird * rotfingerthirdinit * transfingerthird * Vec(0.0, 0.0, 0.0, 1.0);
		}

	}

    int Tid = t * 93;
    for (int i = 0; i < 31; i++) {
      top_data[Tid + i * 3] = x[i][0] ;
      top_data[Tid + i * 3 + 1] = x[i][1] ;
      top_data[Tid + i * 3 + 2] = x[i][2] ;
    }
  }
}

template <typename Dtype>
void libHandModelLayer<Dtype>::Update(std::vector<std::pair<int, int> > Rot, int i, const Dtype* bottom_data, int Bid, Vec x) {
	
	Vec nowx(x[0], x[1], x[2], x[3]);
	for (int r = 0; r < Rot.size(); r++) {
		if (Rot[r].first == -1) //constant matrix trans
		{
			for (int j = 0; j < 47; j++)
			{
				f[i][j] = trans[Rot[r].second] * f[i][j]; //trans latter'			
			}
			nowx = trans[Rot[r].second] * nowx;
		}
		else if (Rot[r].first == -3) //constant matrix rota
		{
			for (int j = 0; j < 47; j++)
			{
				f[i][j] = rota[Rot[r].second] * f[i][j];
			}
			nowx = rota[Rot[r].second] * nowx;
		}
		else if (Rot[r].first == -2) //global translation
		{
			f[i][0] = Matr(3, bottom_data[Bid + 0], bottom_data[Bid + 1], bottom_data[Bid + 2], 1)*nowx + Matr(3, bottom_data[Bid + 0], bottom_data[Bid + 1], bottom_data[Bid + 2], 0)*f[i][0]; //1 w.r.t x
			f[i][1] = Matr(3, bottom_data[Bid + 0], bottom_data[Bid + 1], bottom_data[Bid + 2], 2)*nowx + Matr(3, bottom_data[Bid + 0], bottom_data[Bid + 1], bottom_data[Bid + 2], 0)*f[i][1]; //2 w.r.t y
			f[i][2] = Matr(3, bottom_data[Bid + 0], bottom_data[Bid + 1], bottom_data[Bid + 2], 3)*nowx + Matr(3, bottom_data[Bid + 0], bottom_data[Bid + 1], bottom_data[Bid + 2], 0)*f[i][2]; //2 w.r.t z
			for (int j = 3; j < 47; j++) //other DoF
			{
				f[i][j] = Matr(3, bottom_data[Bid + 0], bottom_data[Bid + 1], bottom_data[Bid + 2], 0)*f[i][j];
			}
			nowx = Matr(3, bottom_data[Bid + 0], bottom_data[Bid + 1], bottom_data[Bid + 2], 0)*nowx;
		}
		else //normal w.r.t x y z rotations
		{
			Matr derivative = Matr(Rot[r].first, bottom_data[Bid + Rot[r].second], 1);
			for (int j = 0; j < 47; j++)
			{
				if (j == Rot[r].second)
				{
					f[i][j] = derivative * nowx + Matr(Rot[r].first, bottom_data[Bid + Rot[r].second], 0) * f[i][j];
				}
				else //irrelevant to the j-th dimension of DoF vector
				{
					f[i][j] = Matr(Rot[r].first, bottom_data[Bid + Rot[r].second], 0) * f[i][j];
				}
			}
			nowx = Matr(Rot[r].first, bottom_data[Bid + Rot[r].second], 0) * nowx;
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
			int Bid = t * 47;

			for (int i = 0; i < 31; i++)
			{
				for (int j = 0; j < 47; j++)
				{
					f[i][j].V[0] = f[i][j].V[1] = f[i][j].V[2] = f[i][j].V[3] = 0.0; //crucial
				}
			}
			//BP palm center      
			std::vector<std::pair<int, int> > Rot;
			Rot.push_back(std::make_pair(-3, 0));
			Rot.push_back(std::make_pair(-2, 0)); //global translation x y z	
			Rot.push_back(std::make_pair(2, 5));
			Rot.push_back(std::make_pair(1, 4));
			Rot.push_back(std::make_pair(0, 3));
			for (int i = 24; i < 25; i++) {
				Update(Rot, i, bottom_data, Bid, Vec(0, 0, 0, 1.0));
			}

			//BP palm root left
			Rot.clear();
			Rot.push_back(std::make_pair(-1, 0));
			Rot.push_back(std::make_pair(-3, 1));
			Rot.push_back(std::make_pair(2, 8));
			Rot.push_back(std::make_pair(1, 7));
			Rot.push_back(std::make_pair(0, 6));
			Rot.push_back(std::make_pair(-3, 0));
			Rot.push_back(std::make_pair(-2, 0));
			Rot.push_back(std::make_pair(2, 5));
			Rot.push_back(std::make_pair(1, 4));
			Rot.push_back(std::make_pair(0, 3));
			for (int i = 25; i < 26; i++) {
				Update(Rot, i, bottom_data, Bid, Vec(0, 0, 0, 1));
			}

			//BP palm root middle
			Rot.clear();
			Rot.push_back(std::make_pair(-1, 1));
			Rot.push_back(std::make_pair(-3, 2));
			Rot.push_back(std::make_pair(2, 11));
			Rot.push_back(std::make_pair(1, 10));
			Rot.push_back(std::make_pair(0, 9));
			Rot.push_back(std::make_pair(-3, 0));
			Rot.push_back(std::make_pair(-2, 0));
			Rot.push_back(std::make_pair(2, 5));
			Rot.push_back(std::make_pair(1, 4));
			Rot.push_back(std::make_pair(0, 3));
			for (int i = 26; i < 27; i++) {
				Update(Rot, i, bottom_data, Bid, Vec(0, 0, 0, 1));
			}
			//

			//BP palm root right
			Rot.clear();
			Rot.push_back(std::make_pair(-1, 2));
			Rot.push_back(std::make_pair(-3, 3));
			Rot.push_back(std::make_pair(2, 14));
			Rot.push_back(std::make_pair(1, 13));
			Rot.push_back(std::make_pair(0, 12));
			Rot.push_back(std::make_pair(-3, 0));
			Rot.push_back(std::make_pair(-2, 0));
			Rot.push_back(std::make_pair(2, 5));
			Rot.push_back(std::make_pair(1, 4));
			Rot.push_back(std::make_pair(0, 3));
			for (int i = 27; i < 28; i++) {
				Update(Rot, i, bottom_data, Bid, Vec(0, 0, 0, 1));
			}
			//      

			//BP thumb
			Rot.clear();
			Rot.push_back(std::make_pair(-1, 3));
			Rot.push_back(std::make_pair(-3, 4));
			Rot.push_back(std::make_pair(2, 16));
			Rot.push_back(std::make_pair(1, 15));
			Rot.push_back(std::make_pair(-1, 2));
			Rot.push_back(std::make_pair(-3, 3));
			Rot.push_back(std::make_pair(2, 14));
			Rot.push_back(std::make_pair(1, 13));
			Rot.push_back(std::make_pair(0, 12));
			Rot.push_back(std::make_pair(-3, 0));
			Rot.push_back(std::make_pair(-2, 0));
			Rot.push_back(std::make_pair(2, 5));
			Rot.push_back(std::make_pair(1, 4));
			Rot.push_back(std::make_pair(0, 3));

			for (int i = 28; i < 29; i++) {
				Update(Rot, i, bottom_data, Bid, Vec(0, 0, 0, 1));
			}

			Rot.clear();
			Rot.push_back(std::make_pair(-1, 4));
			Rot.push_back(std::make_pair(-3, 5));
			Rot.push_back(std::make_pair(2, 17));
			Rot.push_back(std::make_pair(-1, 3));
			Rot.push_back(std::make_pair(-3, 4));
			Rot.push_back(std::make_pair(2, 16));
			Rot.push_back(std::make_pair(1, 15));
			Rot.push_back(std::make_pair(-1, 2));
			Rot.push_back(std::make_pair(-3, 3));
			Rot.push_back(std::make_pair(2, 14));
			Rot.push_back(std::make_pair(1, 13));
			Rot.push_back(std::make_pair(0, 12));
			Rot.push_back(std::make_pair(-3, 0));
			Rot.push_back(std::make_pair(-2, 0));
			Rot.push_back(std::make_pair(2, 5));
			Rot.push_back(std::make_pair(1, 4));
			Rot.push_back(std::make_pair(0, 3));
			for (int i = 29; i < 30; i++) {
				Update(Rot, i, bottom_data, Bid, Vec(0, 0, 0, 1));
			}

			Rot.clear();
			Rot.push_back(std::make_pair(-1, 5));
			Rot.push_back(std::make_pair(-3, 6));
			Rot.push_back(std::make_pair(2, 18));
			Rot.push_back(std::make_pair(-1, 4));
			Rot.push_back(std::make_pair(-3, 5));
			Rot.push_back(std::make_pair(2, 17));
			Rot.push_back(std::make_pair(-1, 3));
			Rot.push_back(std::make_pair(-3, 4));
			Rot.push_back(std::make_pair(2, 16));
			Rot.push_back(std::make_pair(1, 15));
			Rot.push_back(std::make_pair(-1, 2));
			Rot.push_back(std::make_pair(-3, 3));
			Rot.push_back(std::make_pair(2, 14));
			Rot.push_back(std::make_pair(1, 13));
			Rot.push_back(std::make_pair(0, 12));
			Rot.push_back(std::make_pair(-3, 0));
			Rot.push_back(std::make_pair(-2, 0));
			Rot.push_back(std::make_pair(2, 5));
			Rot.push_back(std::make_pair(1, 4));
			Rot.push_back(std::make_pair(0, 3));
			for (int i = 30; i < 31; i++) {
				Update(Rot, i, bottom_data, Bid, Vec(0, 0, 0, 1));
			}


			//BP Bone 1-4
			
			for (int k = 0; k < 4; k++)
			{
				Rot.clear();
				Rot.push_back(std::make_pair(-1, 6 + k));
				Rot.push_back(std::make_pair(-3, 7 + k));
				Rot.push_back(std::make_pair(2, 21 + 3 * k));
				Rot.push_back(std::make_pair(1, 20 + 3 * k));
				Rot.push_back(std::make_pair(0, 19 + 3 * k));
				Rot.push_back(std::make_pair(-3, 0));
				Rot.push_back(std::make_pair(-2, 0));
				Rot.push_back(std::make_pair(2, 5));
				Rot.push_back(std::make_pair(1, 4));
				Rot.push_back(std::make_pair(0, 3));
				for (int i = 20 + k; i < 21 + k; i++) {
					Update(Rot, i, bottom_data, Bid, Vec(0, 0, 0, 1));
				}
			}

			for (int k = 0; k < 4; k++)
			{
				//first
				Rot.clear();
				Rot.push_back(std::make_pair(-1, 10 + 5 * k));
				Rot.push_back(std::make_pair(-3, 11 + 3 * k));
				Rot.push_back(std::make_pair(2, 32 + 4 * k));
				Rot.push_back(std::make_pair(0, 31 + 4 * k));
				Rot.push_back(std::make_pair(-1, 6 + k));
				Rot.push_back(std::make_pair(-3, 7 + k));
				Rot.push_back(std::make_pair(2, 21 + 3 * k));
				Rot.push_back(std::make_pair(1, 20 + 3 * k));
				Rot.push_back(std::make_pair(0, 19 + 3 * k));
				Rot.push_back(std::make_pair(-3, 0));
				Rot.push_back(std::make_pair(-2, 0));
				Rot.push_back(std::make_pair(2, 5));
				Rot.push_back(std::make_pair(1, 4));
				Rot.push_back(std::make_pair(0, 3));
				for (int i = 4 + 5 * k; i < 4 + 5 * k + 1; i++)
				{
					Update(Rot, i, bottom_data, Bid, Vec(0, 0, 0, 1));
				}

				//second
				Rot.clear();
				Rot.push_back(std::make_pair(-1, 11 + 5 * k));
				Rot.push_back(std::make_pair(-3, 12 + 3 * k));
				Rot.push_back(std::make_pair(0, 33 + 4 * k));
				Rot.push_back(std::make_pair(-1, 10 + 5 * k));
				Rot.push_back(std::make_pair(-3, 11 + 3 * k));
				Rot.push_back(std::make_pair(2, 32 + 4 * k));
				Rot.push_back(std::make_pair(0, 31 + 4 * k));
				Rot.push_back(std::make_pair(-1, 6 + k));
				Rot.push_back(std::make_pair(-3, 7 + k));
				Rot.push_back(std::make_pair(2, 21 + 3 * k));
				Rot.push_back(std::make_pair(1, 20 + 3 * k));
				Rot.push_back(std::make_pair(0, 19 + 3 * k));
				Rot.push_back(std::make_pair(-3, 0));
				Rot.push_back(std::make_pair(-2, 0));
				Rot.push_back(std::make_pair(2, 5));
				Rot.push_back(std::make_pair(1, 4));
				Rot.push_back(std::make_pair(0, 3));
				for (int i = 3 + 5 * k; i < 3 + 5 * k + 1; i++) {
					Update(Rot, i, bottom_data, Bid, Vec(0, 0, 0, 1));
				}

				Rot.clear();
				Rot.push_back(std::make_pair(-1, 12 + 5 * k));
				Rot.push_back(std::make_pair(-1, 11 + 5 * k));
				Rot.push_back(std::make_pair(-3, 12 + 3 * k));
				Rot.push_back(std::make_pair(0, 33 + 4 * k));
				Rot.push_back(std::make_pair(-1, 10 + 5 * k));
				Rot.push_back(std::make_pair(-3, 11 + 3 * k));
				Rot.push_back(std::make_pair(2, 32 + 4 * k));
				Rot.push_back(std::make_pair(0, 31 + 4 * k));
				Rot.push_back(std::make_pair(-1, 6 + k));
				Rot.push_back(std::make_pair(-3, 7 + k));
				Rot.push_back(std::make_pair(2, 21 + 3 * k));
				Rot.push_back(std::make_pair(1, 20 + 3 * k));
				Rot.push_back(std::make_pair(0, 19 + 3 * k));
				Rot.push_back(std::make_pair(-3, 0));
				Rot.push_back(std::make_pair(-2, 0));
				Rot.push_back(std::make_pair(2, 5));
				Rot.push_back(std::make_pair(1, 4));
				Rot.push_back(std::make_pair(0, 3));
				for (int i = 2 + 5 * k; i < 2 + 5 * k + 1; i++) {
					Update(Rot, i, bottom_data, Bid, Vec(0, 0, 0, 1));
				}

				//third
				Rot.clear();
				Rot.push_back(std::make_pair(-1, 13 + 5 * k));
				Rot.push_back(std::make_pair(-3, 13 + 3 * k));
				Rot.push_back(std::make_pair(0, 34 + 4 * k));
				Rot.push_back(std::make_pair(-1, 12 + 5 * k));
				Rot.push_back(std::make_pair(-1, 11 + 5 * k));
				Rot.push_back(std::make_pair(-3, 12 + 3 * k));
				Rot.push_back(std::make_pair(0, 33 + 4 * k));
				Rot.push_back(std::make_pair(-1, 10 + 5 * k));
				Rot.push_back(std::make_pair(-3, 11 + 3 * k));
				Rot.push_back(std::make_pair(2, 32 + 4 * k));
				Rot.push_back(std::make_pair(0, 31 + 4 * k));
				Rot.push_back(std::make_pair(-1, 6 + k));
				Rot.push_back(std::make_pair(-3, 7 + k));
				Rot.push_back(std::make_pair(2, 21 + 3 * k));
				Rot.push_back(std::make_pair(1, 20 + 3 * k));
				Rot.push_back(std::make_pair(0, 19 + 3 * k));
				Rot.push_back(std::make_pair(-3, 0));
				Rot.push_back(std::make_pair(-2, 0));
				Rot.push_back(std::make_pair(2, 5));
				Rot.push_back(std::make_pair(1, 4));
				Rot.push_back(std::make_pair(0, 3));
				for (int i = 1 + 5 * k; i < 1 + 5 * k + 1; i++) {
					Update(Rot, i, bottom_data, Bid, Vec(0, 0, 0, 1));
				}

				Rot.clear();
				Rot.push_back(std::make_pair(-1, 14 + 5 * k));
				Rot.push_back(std::make_pair(-1, 13 + 5 * k));
				Rot.push_back(std::make_pair(-3, 13 + 3 * k));
				Rot.push_back(std::make_pair(0, 34 + 4 * k));
				Rot.push_back(std::make_pair(-1, 12 + 5 * k));
				Rot.push_back(std::make_pair(-1, 11 + 5 * k));
				Rot.push_back(std::make_pair(-3, 12 + 3 * k));
				Rot.push_back(std::make_pair(0, 33 + 4 * k));
				Rot.push_back(std::make_pair(-1, 10 + 5 * k));
				Rot.push_back(std::make_pair(-3, 11 + 3 * k));
				Rot.push_back(std::make_pair(2, 32 + 4 * k));
				Rot.push_back(std::make_pair(0, 31 + 4 * k));
				Rot.push_back(std::make_pair(-1, 6 + k));
				Rot.push_back(std::make_pair(-3, 7 + k));
				Rot.push_back(std::make_pair(2, 21 + 3 * k));
				Rot.push_back(std::make_pair(1, 20 + 3 * k));
				Rot.push_back(std::make_pair(0, 19 + 3 * k));
				Rot.push_back(std::make_pair(-3, 0));
				Rot.push_back(std::make_pair(-2, 0));
				Rot.push_back(std::make_pair(2, 5));
				Rot.push_back(std::make_pair(1, 4));
				Rot.push_back(std::make_pair(0, 3));
				for (int i = 0 + 5 * k; i < 0 + 5 * k + 1; i++) {
					Update(Rot, i, bottom_data, Bid, Vec(0, 0, 0, 1));
				}
			}
			


			for (int j = 0; j < 47; j++) {
				bottom_diff[Bid + j] = 0;
				for (int i = 0; i < 31; i++) {
					int Tid = t * 93 + i * 3;
					bottom_diff[Bid + j] += f[i][j][0] * top_diff[Tid] + f[i][j][1] * top_diff[Tid + 1] + f[i][j][2] * top_diff[Tid + 2];
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
