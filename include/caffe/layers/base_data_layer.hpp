#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  // 构造函数，传入参数为solver.prototxt文件
  explicit BaseDataLayer(const LayerParameter& param);    // LayerParameter: /src/caffe/proto/caffe.proto Line310
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  //
  // 该虚函数实现了一般data_layer的功能，能够调用DataLayerSetUp来完成具体的data_layer的设置
  // 只能被BasePrefetchingDataLayer类来重载
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  // 数据层可以被其他的solver共享
  virtual inline bool ShareInParallel() const { return true; }
  //层数据设置，具体要求的data_layer要重载这个函数来具体实现
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  // 数据层没有bottoms
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  
  // 虚函数由子类具体实现
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 protected:
  TransformationParameter transform_param_;
  shared_ptr<DataTransformer<Dtype> > data_transformer_;   // DataTransformer类的智能指针，DataTransformer类主要负责对数据进行预处理
  bool output_labels_;
};

//两个blob类的对象，数据与标签
template <typename Dtype>
class Batch {
 public:
  Blob<Dtype> data_, label_;
};


template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
     
  // 该虚函数实现了一般data_layer的功能，能够调用DataLayerSetUp来完成具体的data_layer的设置
  // 该函数不能被重载
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;

 protected:
  virtual void InternalThreadEntry();                  // 执行线程函数
  virtual void load_batch(Batch<Dtype>* batch) = 0;    // 加载batch

  Batch<Dtype> prefetch_[PREFETCH_COUNT];
  BlockingQueue<Batch<Dtype>*> prefetch_free_;         // 从prefetch_free_队列取数据结构，填充数据结构放到prefetch_full_队列 
  BlockingQueue<Batch<Dtype>*> prefetch_full_;         // 从prefetch_full_队列取数据，使用数据，清空数据结构，放到prefetch_free_队列

  Blob<Dtype> transformed_data_;                       // 转换过的blob数据,中间变量用来辅助图像变换
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
