#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)    
    : Layer<Dtype>(param),                           // 先用param初始化父类Layer
      transform_param_(param.transform_param()) {    // 再初始化transform_param_ ： Blob<Dtype> transformed_data_ 转换过的blob数据,中间变量用来辅助图像变换
}                                                    // optional TransformationParameter transform_param = 100;

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {                             // 判断是否有labels， == 1，false
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  data_transformer_.reset(                           // 创建DataTransformer类的智能指针，用来预处理数据
      new DataTransformer<Dtype>(transform_param_, this->phase_));  // shared_ptr<DataTransformer<Dtype> > data_transformer_;
  data_transformer_->InitRand();                     // 生成随机数据种子
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);                       // 层数据设置
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_() {
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_free_.push(&prefetch_[i]);
  }
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
    
  // 在开启prefetch线程之前，调用cpu_data和gpu_data，这样主线程正在运行时，prefetch线程避免同时调用cudaMalloc
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_[i].data_.mutable_cpu_data();          // 依次给队列中每个batch的数据blob分配cpu内存
    if (this->output_labels_) {
      prefetch_[i].label_.mutable_cpu_data();       // 依次分配每个每个batch的标签blob分配cpu内存
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
      prefetch_[i].data_.mutable_gpu_data();        // 依次给队列中每个batch的数据blob分配gpu内存
      if (this->output_labels_) {
        prefetch_[i].label_.mutable_gpu_data();     // 依次分配每个每个batch的标签blob分配gpu内存
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();                            // 启动内部读取数据线程
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {   // 如果有空闲线程，让该线程去取数据
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      Batch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
        top[1]->mutable_cpu_data());
  }

  prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
