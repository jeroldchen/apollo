/******************************************************************************
 * Copyright 2020 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include "modules/perception/inference/tensorrt/plugins/rcnn_proposal_plugin.h"

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include "modules/perception/inference/tensorrt/plugins/kernels.h"

#define DEBUG 1

#if DEBUG
#include <fstream>
#include <sys/stat.h>
#endif

namespace apollo {
namespace perception {
namespace inference {

// nthreads = num_rois
__global__ void get_rois_nums_kernel(const int nthreads, const float *rois,
                                     int *batch_rois_nums) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    int batch_id = (int)rois[index * 5];
    if (batch_id >= 0) {
      atomicAdd(&batch_rois_nums[batch_id], 1);
    }
  }
}

// bbox_pred dims: [num_rois, box_len, num_class]
// out_bbox_pred dims: [num_rois, num_class, box_len]
__global__ void transpose_bbox_pred_kernel(const int nthreads,
                                           const float *bbox_pred,
                                           const int box_len,
                                           const int num_class,
                                           float *out_bbox_pred) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    int roi_id = index / num_class / box_len;
    int class_id = (index / box_len) % num_class;
    int feature_id = index % box_len;

    int in_index =
        roi_id * box_len * num_class + feature_id * num_class + class_id;
    out_bbox_pred[index] = bbox_pred[in_index];
  }
}

// bbox_pred dims: [num_box, num_class+1, 4],
// scores dims: [num_box, num_class+1],
// out_bbox_pred dims: [num_box, 4]
// out_scores dims: [num_box]
__global__ void get_max_score_kernel(const int nthreads, const float *bbox_pred,
                                     const float *scores, const int num_class,
                                     const float threshold_objectness,
                                     const float *class_thresholds,
                                     float *out_bbox_pred, float *out_scores,
                                     int *filter_count) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= nthreads) {
    return;
  }

  int box_id = index;
  if ((1.0f - scores[box_id * (num_class + 1)]) < threshold_objectness) {
    return;
  }

  float score_max = -FLT_MAX;
  int cls_max = -1;
  for (int c = 0; c < num_class; ++c) {
    float score =
        scores[box_id * (num_class + 1) + c + 1] - class_thresholds[c];
    if (score > score_max) {
      score_max = score;
      cls_max = c;
    }
  }
  if (score_max < 0) {
    return;
  } else {
    int counter = atomicAdd(filter_count, 1);
    int box_cls_id = box_id * (num_class + 1) + cls_max + 1;
    for (int i = 0; i < 4; ++i) {
      out_bbox_pred[counter * 4 + i] = bbox_pred[box_cls_id * 4 + i];
    }
    out_scores[counter] = scores[box_cls_id];
  }
}

int RCNNProposalPlugin::enqueue(int batchSize, const void *const *inputs,
                                void **outputs, void *workspace,
                                cudaStream_t stream) {
  // cls_score_softmax dims: [num_rois, 4, 1, 1]
  const float* cls_score_softmax = reinterpret_cast<const float*>(inputs[0]);
  // TODO(cjh): num cls, box_dim or reverse?
  // bbox_pred dims: [num_rois, 4 * 4 (num_class * box_dim), 1, 1]
  const float* bbox_pred = reinterpret_cast<const float*>(inputs[1]);
  // rois dims: [num_rois, 5, 1, 1]
  const float* rois = reinterpret_cast<const float*>(inputs[2]);
  // im_info dims: [N, 6, 1, 1]
  const float* im_info = reinterpret_cast<const float*>(inputs[3]);
  // output dims: [num_result_box, 9] (axis-1: batch_id, x1, y1, x2, y2,
  //   unknown_score, class1_score, class2_score, class3_score)
  float *result_boxes = reinterpret_cast<float*>(outputs[0]);

  int cls_score_softmax_size = num_rois_ * 4;
  int bbox_pred_size = num_rois_ * 4 * 4;
  int output_size = batchSize * top_n_ * out_channel_;

  thrust::fill(thrust::device, result_boxes, result_boxes + size_t(output_size), -1.0f);

#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
  float *host_bbox_pred = new float[bbox_pred_size]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_bbox_pred, bbox_pred,
                                  bbox_pred_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::ofstream f_bbox_pred;
  std::string f_bbox_pred_str = "/apollo/debug/rcnn_bbox_pred.txt";
  struct stat buffer;
  if (stat(f_bbox_pred_str.c_str(), &buffer) != 0) {
    f_bbox_pred.open(f_bbox_pred_str.c_str());
    for (int i = 0; i < bbox_pred_size; ++i) {
      if (i % 4 == 0) {
        f_bbox_pred << "\n";
      }
      f_bbox_pred << host_bbox_pred[i] << " ";
    }
  }
  delete[] host_bbox_pred;
#endif
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
  float *host_cls_score_softmax = new float[cls_score_softmax_size]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_cls_score_softmax, cls_score_softmax,
                                  cls_score_softmax_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::ofstream f_cls_score_softmax;
  std::string f_cls_score_softmax_str = "/apollo/debug/rcnn_cls_score_softmax.txt";
  struct stat f_cls_score_softmax_buffer;
  if (stat(f_cls_score_softmax_str.c_str(), &f_cls_score_softmax_buffer) != 0) {
    f_cls_score_softmax.open(f_cls_score_softmax_str.c_str());
    for (int i = 0; i < cls_score_softmax_size; ++i) {
      if (i % 4 == 0) {
        f_cls_score_softmax << "\n";
      }
      f_cls_score_softmax << host_cls_score_softmax[i] << " ";
    }
  }
  delete[] host_cls_score_softmax;
#endif

  float *host_im_info = new float[batchSize * 6]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_im_info, im_info,
                             batchSize * 6 * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  float origin_height = host_im_info[0];
  float origin_width = host_im_info[1];
  float scale = host_im_info[2];

  int nthreads, block_size;

  // TODO(chenjiahao): filter roi that has img_id == -1 at first
  // TODO(chenjiahao): save index all the time to retrieve the data

  float *host_thresholds = new float[num_class_];
  for (int i = 0; i < num_class_; ++i) {
    host_thresholds[i] = thresholds_[i];
  }
  float *dev_thresholds;
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_thresholds),
                             num_class_ * sizeof(float)));
  BASE_CUDA_CHECK(cudaMemcpyAsync(dev_thresholds, host_thresholds,
                                  num_class_ * sizeof(float),
                                  cudaMemcpyHostToDevice, stream));

  // Transpose bbox_pred from [, box_len, num_class] to [, num_class, box_len]
//  float *transposed_bbox_pred;
//  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&transposed_bbox_pred),
//                             bbox_pred_size * sizeof(float)));
//  nthreads = bbox_pred_size;
//  block_size = DIVUP(nthreads, thread_size_);
//  transpose_bbox_pred_kernel<<<block_size, thread_size_, 0, stream>>>(nthreads,
//      bbox_pred, 4, 4, transposed_bbox_pred);

  // Normalize bbox_pred
  float *dev_bbox_mean, *dev_bbox_std;
  float *norm_bbox_pred;
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_bbox_mean), 4 * sizeof(float)));
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_bbox_std), 4 * sizeof(float)));
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&norm_bbox_pred),
                             bbox_pred_size * sizeof(float)));
  BASE_CUDA_CHECK(cudaMemcpyAsync(dev_bbox_mean, bbox_mean_, 4 * sizeof(float), cudaMemcpyHostToDevice, stream));
  BASE_CUDA_CHECK(cudaMemcpyAsync(dev_bbox_std, bbox_std_, 4 * sizeof(float), cudaMemcpyHostToDevice, stream));
//  BASE_CUDA_CHECK(cudaMemcpyAsync(norm_bbox_pred, transposed_bbox_pred,
  BASE_CUDA_CHECK(cudaMemcpyAsync(norm_bbox_pred, bbox_pred,
      bbox_pred_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
  nthreads = bbox_pred_size;
  block_size = DIVUP(nthreads, thread_size_);
  repeatedly_mul_cuda(block_size, thread_size_, 0, stream, nthreads,
      norm_bbox_pred, norm_bbox_pred, dev_bbox_std, 4);
  repeatedly_add_cuda(block_size, thread_size_, 0, stream, nthreads,
      norm_bbox_pred, norm_bbox_pred, dev_bbox_mean, 4);

  // Slice and rescale rois
  int slice_axis[4] = {1, 2, 3, 4};
  int *dev_slice_axis;
  float *sliced_rois;
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_slice_axis),
                             4 * sizeof(int)));
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sliced_rois),
                             num_rois_ * 4 * sizeof(float)));
  BASE_CUDA_CHECK(cudaMemcpyAsync(dev_slice_axis, slice_axis, 4 * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  nthreads = num_rois_ * 4;
  block_size = DIVUP(nthreads, thread_size_);
  slice2d_cuda(block_size, thread_size_, 0, stream, nthreads, rois,
      sliced_rois, dev_slice_axis, 4, 5);
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
  float *host_sliced_rois = new float[num_rois_ * 4]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_sliced_rois, sliced_rois,
                                  num_rois_ * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::ofstream f_sliced_rois;
  std::string f_sliced_rois_str = "/apollo/debug/rcnn_sliced_rois.txt";
  struct stat sliced_rois_buffer;
  if (stat(f_sliced_rois_str.c_str(), &sliced_rois_buffer) != 0) {
    f_sliced_rois.open(f_sliced_rois_str.c_str());
    for (int i = 0; i < num_rois_ * 4; ++i) {
      if (i % 4 == 0) {
        f_sliced_rois << "\n";
      }
      f_sliced_rois << host_sliced_rois[i] << " ";
    }
  }
  delete[] host_sliced_rois;
#endif

//  float *dev_scale;
//  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_scale),
//                            sizeof(float)));
//  BASE_CUDA_CHECK(cudaMemsetAsync(dev_scale, scale, sizeof(float), stream));
//  nthreads = num_rois_ * 4;
//  block_size = DIVUP(nthreads, thread_size_);
//  repeatedly_div_cuda(block_size, thread_size_, 0, stream, nthreads,
//      sliced_rois, sliced_rois, dev_scale, 1);

//  thrust::transform(thrust::device, sliced_rois,
//                    sliced_rois + size_t(num_rois_ * 4),
//                    thrust::make_constant_iterator(1 / scale),
//                    sliced_rois,
//                    thrust::multiplies<float>());

#if DEBUG
//  BASE_CUDA_CHECK(cudaDeviceSynchronize());
//  float *host_scale_rois = new float[num_rois_ * 4]();
//  BASE_CUDA_CHECK(cudaMemcpyAsync(host_scale_rois, sliced_rois,
//                                  num_rois_ * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
//  std::ofstream f_scale_rois;
//  std::string f_scale_rois_str = "/apollo/debug/rcnn_scale_rois.txt";
//  struct stat scale_rois_buffer;
//  if (stat(f_scale_rois_str.c_str(), &scale_rois_buffer) != 0) {
//    f_scale_rois.open(f_scale_rois_str.c_str());
//    for (int i = 0; i < num_rois_ * 4; ++i) {
//      if (i % 4 == 0) {
//        f_scale_rois << "\n";
//      }
//      f_scale_rois << host_scale_rois[i] << " ";
//    }
//  }
//  delete[] host_scale_rois;
#endif

  // Decode bbox
  float *decoded_bbox_pred;
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&decoded_bbox_pred),
                             bbox_pred_size * sizeof(float)));
  BASE_CUDA_CHECK(cudaMemsetAsync(decoded_bbox_pred, 0, bbox_pred_size * sizeof(float), stream));
  nthreads = num_rois_ * 4;
  block_size = DIVUP(nthreads, thread_size_);
  bbox_transform_inv_cuda(block_size, thread_size_, 0, stream, nthreads,
      sliced_rois, norm_bbox_pred, num_rois_, 4, decoded_bbox_pred);
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
  float *host_bbox = new float[bbox_pred_size]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_bbox, decoded_bbox_pred,
                                  bbox_pred_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::ofstream f_bbox;
  std::string f_bbox_str = "/apollo/debug/rcnn_bbox.txt";
  struct stat bbox_buffer;
  if (stat(f_bbox_str.c_str(), &bbox_buffer) != 0) {
    f_bbox.open(f_bbox_str.c_str());
    for (int i = 0; i < bbox_pred_size; ++i) {
      if (i % 4 == 0) {
        f_bbox << "\n";
      }
      f_bbox << host_bbox[i] << " ";
    }
  }
  delete[] host_bbox;
#endif

  // Refine boxes that are out of map
  if (refine_out_of_map_bbox_) {
    nthreads = bbox_pred_size;
    block_size = DIVUP(nthreads, thread_size_);
    clip_boxes_cuda(block_size, thread_size_, 0, stream, nthreads,
        decoded_bbox_pred, origin_height, origin_width);
  }
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
  float *host_refined_bbox = new float[bbox_pred_size]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_refined_bbox, decoded_bbox_pred,
                                  bbox_pred_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::ofstream f_refined_bbox;
  std::string f_refined_bbox_str = "/apollo/debug/rcnn_refined_bbox.txt";
  struct stat f_refined_bbox_buffer;
  if (stat(f_refined_bbox_str.c_str(), &f_refined_bbox_buffer) != 0) {
    f_refined_bbox.open(f_refined_bbox_str.c_str());
    for (int i = 0; i < bbox_pred_size; ++i) {
      if (i % 4 == 0) {
        f_refined_bbox << "\n";
      }
      f_refined_bbox << host_refined_bbox[i] << " ";
    }
  }
  delete[] host_refined_bbox;
#endif

  // Separate data by batch_id
  int *batch_rois_nums = new int[batchSize]();
  int *dev_batch_rois_nums;
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_batch_rois_nums),
                             batchSize * sizeof(int)));
  BASE_CUDA_CHECK(cudaMemsetAsync(dev_batch_rois_nums, 0, batchSize * sizeof(int), stream));
  nthreads = num_rois_;
  block_size = DIVUP(nthreads, thread_size_);
  get_rois_nums_kernel<<<block_size, thread_size_, 0, stream>>>(nthreads, rois,
      dev_batch_rois_nums);
  BASE_CUDA_CHECK(cudaMemcpyAsync(batch_rois_nums, dev_batch_rois_nums,
                             batchSize * sizeof(int), cudaMemcpyDeviceToHost, stream));

  float *max_bbox, *max_score;
  int *max_filtered_count;
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&max_bbox),
                             max_candidate_n_ * 4 * sizeof(float)));
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&max_score),
                             max_candidate_n_ * sizeof(float)));
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&max_filtered_count),
                             sizeof(int)));

  float *filtered_bbox, *filtered_score;
  int *filtered_count;
  int host_filtered_count;
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&filtered_bbox),
                             max_candidate_n_ * 4 * sizeof(float)));
  BASE_CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&filtered_score),
                 max_candidate_n_ * sizeof(float)));
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&filtered_count),
                             sizeof(int)));

  int *sorted_indexes;
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sorted_indexes),
                             max_candidate_n_ * sizeof(int)));

  float *pre_nms_bbox, *pre_nms_score;
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pre_nms_bbox),
                             max_candidate_n_ * 4 * sizeof(float)));
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pre_nms_score),
                             max_candidate_n_ * sizeof(float)));

  int cur_ptr = 0;
  acc_box_num_ = 0;
  for (int batch_id = 0; batch_id < batchSize; ++batch_id) {
    // TODO(chenjiahao): replace 300 with param
    cur_ptr = batch_id * 300;
    BASE_CUDA_CHECK(cudaMemsetAsync(max_bbox, 0,
                                    max_candidate_n_ * 4 * sizeof(float), stream));
    BASE_CUDA_CHECK(cudaMemsetAsync(max_score, 0,
                                    max_candidate_n_ * sizeof(float), stream));
    BASE_CUDA_CHECK(cudaMemsetAsync(max_filtered_count, 0, sizeof(int), stream));
    // Get max score among classes and filter with threshold
    nthreads = batch_rois_nums[batch_id];
    block_size = DIVUP(nthreads, thread_size_);
    get_max_score_kernel<<<block_size, thread_size_, 0, stream>>>(
        nthreads, decoded_bbox_pred + size_t(cur_ptr * 16),
        cls_score_softmax + size_t(cur_ptr * 4), 3, threshold_objectness_,
        dev_thresholds, max_bbox, max_score, max_filtered_count);
    int host_max_filtered_count = 0;
    BASE_CUDA_CHECK(cudaMemcpyAsync(&host_max_filtered_count, max_filtered_count,
                                    sizeof(int), cudaMemcpyDeviceToHost, stream));
    if (host_max_filtered_count == 0) {
      continue;
    }

    BASE_CUDA_CHECK(cudaMemsetAsync(filtered_bbox, 0,
                                    max_candidate_n_ * 4 * sizeof(float), stream));
    BASE_CUDA_CHECK(cudaMemsetAsync(filtered_score, 0,
                                    max_candidate_n_ * sizeof(float), stream));
    BASE_CUDA_CHECK(cudaMemsetAsync(filtered_count, 0, sizeof(int), stream));
    // Filter boxes according to min_size_mode
    nthreads = host_max_filtered_count;
    block_size = DIVUP(nthreads, thread_size_);
    filter_boxes_cuda(block_size, thread_size_, 0, stream, nthreads,
                      max_bbox, max_score, host_max_filtered_count, 1,1, 0, 0,
                      min_size_mode_, min_size_h_, min_size_w_,0.0f,
                      filtered_bbox, filtered_score, filtered_count);

    BASE_CUDA_CHECK(cudaMemcpyAsync(&host_filtered_count, filtered_count,
                               sizeof(int), cudaMemcpyDeviceToHost, stream));
    if (host_filtered_count == 0) {
      continue;
    }
#if DEBUG
      BASE_CUDA_CHECK(cudaDeviceSynchronize());
//      AERROR << "host_filtered_count: " << host_filtered_count
//             << " batch_rois_num: " << batch_rois_nums[batch_id];
      float *host_filtered_bbox = new float[max_candidate_n_ * 4]();
      BASE_CUDA_CHECK(cudaMemcpyAsync(host_filtered_bbox, filtered_bbox,
                                      max_candidate_n_ * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
      std::ofstream f_filtered_bbox;
      std::string f_filtered_bbox_str = "/apollo/debug/rcnn_filtered_bbox.txt";
//      struct stat f_filtered_bbox_buffer;
//      if (stat(f_filtered_bbox_str.c_str(), &f_filtered_bbox_buffer) != 0) {
        f_filtered_bbox.open(f_filtered_bbox_str.c_str());
        for (int i = 0; i < max_candidate_n_ * 4; ++i) {
          if (i % 4 == 0) {
            f_filtered_bbox << "\n";
          }
          f_filtered_bbox << host_filtered_bbox[i] << " ";
        }
//      }
      delete[] host_filtered_bbox;
#endif
#if DEBUG
      BASE_CUDA_CHECK(cudaDeviceSynchronize());
      float *host_filtered_score = new float[max_candidate_n_]();
      BASE_CUDA_CHECK(cudaMemcpyAsync(host_filtered_score, filtered_score,
                                      max_candidate_n_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
      std::ofstream f_filtered_score;
      std::string f_filtered_score_str = "/apollo/debug/rcnn_filtered_score.txt";
//      struct stat f_filtered_score_buffer;
//      if (stat(f_filtered_score_str.c_str(), &f_filtered_score_buffer) != 0) {
        f_filtered_score.open(f_filtered_score_str.c_str());
        for (int i = 0; i < max_candidate_n_; ++i) {
          if (i % 4 == 0) {
            f_filtered_score << "\n";
          }
          f_filtered_score << host_filtered_score[i] << " ";
        }
//      }
      delete[] host_filtered_score;
#endif

    // descending sort proposals by score
    thrust::sequence(thrust::device, sorted_indexes,
                     sorted_indexes + host_filtered_count);
    thrust::sort_by_key(thrust::device, filtered_score,
                        filtered_score + size_t(host_filtered_count),
                        sorted_indexes, thrust::greater<float>());

    BASE_CUDA_CHECK(cudaMemsetAsync(pre_nms_bbox, 0,
                                    max_candidate_n_ * 4 * sizeof(float), stream));
    BASE_CUDA_CHECK(cudaMemsetAsync(pre_nms_score, 0,
                                    max_candidate_n_ * sizeof(float), stream));
    // keep max N candidates
    nthreads = std::min(max_candidate_n_, host_filtered_count);
    block_size = DIVUP(nthreads, thread_size_);
    keep_topN_boxes_cuda(block_size, thread_size_, 0, stream, nthreads,
        filtered_bbox, filtered_score, sorted_indexes, filtered_count,
        rpn_proposal_output_score_, max_candidate_n_,
        max_candidate_n_, pre_nms_bbox, pre_nms_score);
#if DEBUG
      BASE_CUDA_CHECK(cudaDeviceSynchronize());
      float *host_pre_nms_bbox = new float[max_candidate_n_ * 4]();
      BASE_CUDA_CHECK(cudaMemcpyAsync(host_pre_nms_bbox, pre_nms_bbox,
                                      max_candidate_n_ * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
      std::ofstream f_pre_nms_bbox;
      std::string f_pre_nms_bbox_str = "/apollo/debug/rcnn_pre_nms_bbox.txt";
      struct stat f_pre_nms_bbox_buffer;
      if (stat(f_pre_nms_bbox_str.c_str(), &f_pre_nms_bbox_buffer) != 0) {
        f_pre_nms_bbox.open(f_pre_nms_bbox_str.c_str());
        for (int i = 0; i < max_candidate_n_ * 4; ++i) {
          if (i % 4 == 0) {
            f_pre_nms_bbox << "\n";
          }
          f_pre_nms_bbox << host_pre_nms_bbox[i] << " ";
        }
      }
      delete[] host_pre_nms_bbox;
#endif
#if DEBUG
      BASE_CUDA_CHECK(cudaDeviceSynchronize());
      float *host_pre_nms_score = new float[max_candidate_n_]();
      BASE_CUDA_CHECK(cudaMemcpyAsync(host_pre_nms_score, pre_nms_score,
                                      max_candidate_n_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
      std::ofstream f_pre_nms_score;
      std::string f_pre_nms_score_str = "/apollo/debug/rcnn_pre_nms_score.txt";
      struct stat f_pre_nms_score_buffer;
      if (stat(f_pre_nms_score_str.c_str(), &f_pre_nms_score_buffer) != 0) {
        f_pre_nms_score.open(f_pre_nms_score_str.c_str());
        for (int i = 0; i < max_candidate_n_; ++i) {
          if (i % 4 == 0) {
            f_pre_nms_score << "\n";
          }
          f_pre_nms_score << host_pre_nms_score[i] << " ";
        }
      }
      delete[] host_pre_nms_score;
#endif

    // NMS
    int cur_filter_count = std::min(host_filtered_count, max_candidate_n_);
    // TODO(cjh): debugging. Need to track the index of scores.
    NmsForward(rpn_proposal_output_score_, cur_filter_count, 4,
               overlap_ratio_, max_candidate_n_, top_n_, batch_id, 1,
               pre_nms_bbox, pre_nms_score,
               result_boxes + size_t(acc_box_num_ * out_channel_),
               &acc_box_num_, stream);
  }

  // TODO(henjiahao): output all classification scores
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
  float *host_bboxes = new float[top_n_ * 9]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_bboxes, result_boxes,
                                  top_n_ * 9 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::ofstream f_bboxes;
  std::string f_bboxes_str = "/apollo/debug/rcnn_bboxes.txt";
  struct stat f_bboxes_buffer;
  if (stat(f_bboxes_str.c_str(), &f_bboxes_buffer) != 0) {
    f_bboxes.open(f_bboxes_str.c_str());
    for (int i = 0; i < top_n_ * 9; ++i) {
      if (i % 9 == 0) {
        f_bboxes << "\n";
      }
      f_bboxes << host_bboxes[i] << " ";
    }
  }
  delete[] host_bboxes;
#endif

  // Free device memory
  BASE_CUDA_CHECK(cudaFree(dev_thresholds));
//  BASE_CUDA_CHECK(cudaFree(transposed_bbox_pred));
  BASE_CUDA_CHECK(cudaFree(dev_bbox_mean));
  BASE_CUDA_CHECK(cudaFree(dev_bbox_std));
  BASE_CUDA_CHECK(cudaFree(norm_bbox_pred));
  BASE_CUDA_CHECK(cudaFree(dev_slice_axis));
  BASE_CUDA_CHECK(cudaFree(sliced_rois));
  BASE_CUDA_CHECK(cudaFree(decoded_bbox_pred));
  BASE_CUDA_CHECK(cudaFree(dev_batch_rois_nums));
  BASE_CUDA_CHECK(cudaFree(max_bbox));
  BASE_CUDA_CHECK(cudaFree(max_score));
  BASE_CUDA_CHECK(cudaFree(max_filtered_count));
  BASE_CUDA_CHECK(cudaFree(filtered_bbox));
  BASE_CUDA_CHECK(cudaFree(filtered_score));
  BASE_CUDA_CHECK(cudaFree(filtered_count));
  BASE_CUDA_CHECK(cudaFree(sorted_indexes));
  BASE_CUDA_CHECK(cudaFree(pre_nms_bbox));
  BASE_CUDA_CHECK(cudaFree(pre_nms_score));

  // Free host memory
  delete[] host_im_info;
  delete[] host_thresholds;
  delete[] batch_rois_nums;
}

}  // namespace inference
}  // namespace perception
}  // namespace apollo