#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

const int kMaxBlobAxes = 32;

namespace caffe {

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Blob {
 public:
  Blob()
       : data_(), diff_(), count_(0), capacity_(0) {}             // 默认构造函数：一个Blob存储了data_ , diff_

  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
  explicit Blob(const int num, const int channels, const int height,
      const int width);
  explicit Blob(const vector<int>& shape);          //用这个函数

  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
  void Reshape(const int num, const int channels, const int height,
      const int width);
  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.   
   * 
   * 内存不够时才会重新分配内存，并且不会释放多余的内存
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is  
   * an error; either Net::Forward or Net::Reshape need to be called to      
   * propagate the new input shape to higher layers. 
   * 
   * reshape一个input blob并且立即调用Net::Backward是错误的
   */
  void Reshape(const vector<int>& shape);
  void Reshape(const BlobShape& shape);
  void ReshapeLike(const Blob& other);
  inline string shape_string() const {
    ostringstream stream;            //输出数据的维度，以空格分隔，最后输出一维维度（total）
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";  //一维维度
    return stream.str();
  }
  inline const vector<int>& shape() const { return shape_; }      //返回shape_:Blob的形状 
  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  inline int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];                    //返回Blob某一维尺寸的大小
  }
  inline int num_axes() const { return shape_.size(); }          //返回数据维度的大小,vector<int> shape_; 一般为4维
  inline int count() const { return count_; }                    //返回数据个数

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
  inline int count(int start_axis, int end_axis) const {         //获取从start维到end维之间的数据个数
    CHECK_LE(start_axis, end_axis);           // CHECK_LE(x,y) ,LE即lower equation,意为小于等于，函数判断是否x小于等于y
    CHECK_GE(start_axis, 0);                  // CHECK_GE(x,y) ,GE即为great equation，意为大于
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }
  /**
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
  inline int count(int start_axis) const {                      //获取start维度到最后的维度之间的数据个数
    return count(start_axis, num_axes());
  }

  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  inline int CanonicalAxisIndex(int axis_index) const {       //支持负数维度索引，负数表示从后往前
    CHECK_GE(axis_index, -num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }

  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.    //弃用，使用shape(0)返回num
  inline int num() const { return LegacyShape(0); }
  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  inline int channels() const { return LegacyShape(1); }
  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  inline int height() const { return LegacyShape(2); }
  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  inline int width() const { return LegacyShape(3); }
  inline int LegacyShape(int index) const {                                //检查blob的维度个数是不是小于
    CHECK_LE(num_axes(), 4)                //检查num_axes()是不是小于4
        << "Cannot use legacy accessors on Blobs with > 4 axes.";
    CHECK_LT(index, 4);                    //CHECK_LT(x,y) ,LT即为lower to ，意为小于，函数判断是否x小于y
    CHECK_GE(index, -4);
    if (index >= num_axes() || index < -num_axes()) {
      // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
      // indexing) -- this special case simulates the one-padding used to fill
      // extraneous axes of legacy blobs.
      return 1;
    }
    return shape(index);                  //调用shape()函数返回某一维数据个数
  }

  inline int offset(const int n, const int c = 0, const int h = 0,        //计算一维线性偏移量
      const int w = 0) const {
    CHECK_GE(n, 0);
    CHECK_LE(n, num());
    CHECK_GE(channels(), 0);
    CHECK_LE(c, channels());
    CHECK_GE(height(), 0);
    CHECK_LE(h, height());
    CHECK_GE(width(), 0);
    CHECK_LE(w, width());
    return ((n * channels() + c) * height() + h) * width() + w;
  }

  inline int offset(const vector<int>& indices) const {                  //参数用vector
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];                //((n * channels() + c) * height() + h) * width() + w;
      }
    }
    return offset;
  }
  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * false 复制data；true 复制diff
   * 
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   * reshape为true则改变blob形状
   */
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);
  
  // 获取在内存下的数据（前向传播用的数据）
  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_data()[offset(n, c, h, w)];
  }
  
  // 获取在内存下的数据（反向传播用的数据）
  inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_diff()[offset(n, c, h, w)];
  }
  
  // 
  inline Dtype data_at(const vector<int>& index) const {
    return cpu_data()[offset(index)];
  }

  inline Dtype diff_at(const vector<int>& index) const {
    return cpu_diff()[offset(index)];
  }
  
  // 返回前向传播数据地址
  inline const shared_ptr<SyncedMemory>& data() const {          
    CHECK(data_);
    return data_;
  }
  
  // 返回反向传播数据地址
  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }

  const Dtype* cpu_data() const;
  void set_cpu_data(Dtype* data);
  const int* gpu_shape() const;
  const Dtype* gpu_data() const;
  const Dtype* cpu_diff() const;
  const Dtype* gpu_diff() const;
  Dtype* mutable_cpu_data();
  Dtype* mutable_gpu_data();
  Dtype* mutable_cpu_diff();
  Dtype* mutable_gpu_diff();
  void Update();                         // 数据更新：blob里的data部分减去diff部分
  void FromProto(const BlobProto& proto, bool reshape = true);        // 从protobuf序列化文件读取blob对象
  void ToProto(BlobProto* proto, bool write_diff = false) const;      // 将对象序列化为protobuf文件

  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  Dtype asum_data() const;                  // 计算data的L1范数
  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  Dtype asum_diff() const;                  // 计算diff的L1范数
  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  Dtype sumsq_data() const;                  // 计算data的L2范数
  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
  Dtype sumsq_diff() const;                  // 计算diff的L2范数

  /// @brief Scale the blob data by a constant factor.
  void scale_data(Dtype scale_factor);       //归一化data数据
  /// @brief Scale the blob diff by a constant factor.
  void scale_diff(Dtype scale_factor);       //归一化diff数据

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareData(const Blob& other);         //与other共享数据，把other的data数据指针传给本blob
  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareDiff(const Blob& other);         //与other共享数据，把other的diff数据指针传给本blob

  bool ShapeEquals(const BlobProto& other);  //判断本blob与other形状是否相等  

 protected:
  shared_ptr<SyncedMemory> data_;        //申请内存存放data，用于正向传播
  shared_ptr<SyncedMemory> diff_;        //申请内存存储diff
  shared_ptr<SyncedMemory> shape_data_;  //老版本存储blob形状
  vector<int> shape_;                    //新版本存储blob形状
  int count_;                            //一维维度，即blob数据个数 = N * C * H * W
  int capacity_;                         //元素个数，即内存最大能存储数据大小
  
  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
