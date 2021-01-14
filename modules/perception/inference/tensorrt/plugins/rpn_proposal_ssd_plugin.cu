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

#include "modules/perception/inference/tensorrt/plugins/rpn_proposal_ssd_plugin.h"

#include <thrust/fill.h>
#include <thrust/sort.h>

#include "modules/perception/inference/tensorrt/plugins/kernels.h"

#define DEBUG 1

#if DEBUG
#include <fstream>
#include <sys/stat.h>
#endif

namespace apollo {
namespace perception {
namespace inference {

// output anchors dims: [H, W, num_anchor_per_point, 4]
__global__ void generate_anchors_kernel(const int height, const int width,
                                        const float anchor_stride,
                                        const int num_anchor_per_point,
                                        const float *anchor_heights,
                                        const float *anchor_widths,
                                        float *anchors) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int num_anchor = height * width * num_anchor_per_point;
  if (index >= num_anchor) {
    return;
  }

//  float anchor_offset = 0.5 * anchor_stride;
  float anchor_offset = 0;
  int pos_index = index / num_anchor_per_point;
  int anchor_id = index % num_anchor_per_point;
  int w_i = pos_index % width;
  int h_i = pos_index / width;

  // center coordinates
  float x_ctr = w_i * anchor_stride + anchor_offset;
  float y_ctr = h_i * anchor_stride + anchor_offset;

  // TODO(cjh): testing
//  float x_min = x_ctr - 0.5 * anchor_widths[anchor_id];
//  float y_min = y_ctr - 0.5 * anchor_heights[anchor_id];
//  float x_max = x_ctr + 0.5 * anchor_widths[anchor_id] - 1;
//  float y_max = y_ctr + 0.5 * anchor_heights[anchor_id] - 1;
  float x_min = x_ctr - 0.5 * (anchor_widths[anchor_id] - 1);
  float y_min = y_ctr - 0.5 * (anchor_heights[anchor_id] - 1);
  float x_max = x_ctr + 0.5 * (anchor_widths[anchor_id] - 1);
  float y_max = y_ctr + 0.5 * (anchor_heights[anchor_id] - 1);

  anchors[index * 4] = x_min;
  anchors[index * 4 + 1] = y_min;
  anchors[index * 4 + 2] = x_max;
  anchors[index * 4 + 3] = y_max;
}


// in_boxes dims: [N, num_box_per_point * 4, H, W],
// out_boxes dims: [N, H * W * num_box_per_point， 4]
template<typename Dtype>
__global__ void reshape_boxes_kernel(const int nthreads, const Dtype *in_boxes,
                                     const int height, const int width,
                                     const int num_box_per_point,
                                     Dtype *out_boxes) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    int num_point = height * width;

    int batch_id = index / num_point / num_box_per_point / 4;
    int feature_id = index % 4;
    int box_id = (index / 4) % num_box_per_point;
    int point_id = (index / num_box_per_point / 4) % num_point;

//    int in_index = ((batch_id * 4 + feature_id) * num_box_per_point + box_id) * num_point + point_id;
    int in_index = ((batch_id * num_box_per_point + box_id) * 4 + feature_id) * num_point + point_id;
    out_boxes[index] = in_boxes[in_index];
  }
}

// in_scores dims: [N, 2 * num_box_per_point, H, W],
// out_scores dims: [N, H * W * num_box_per_point, 2]
template<typename Dtype>
__global__ void reshape_scores_kernel(const int nthreads,
                                      const Dtype *in_scores,
                                      const int height, const int width,
                                      const int num_box_per_point,
                                      Dtype *out_scores) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    int num_point = height * width;

    int batch_id = index / num_point / num_box_per_point / 2;
    int class_id = index % 2;
    int box_id = (index / 2) % num_box_per_point;
    int point_id = (index / num_box_per_point / 2) % num_point;

    int in_index = ((batch_id * 2 + class_id) * num_box_per_point + box_id) * num_point + point_id;
    out_scores[index] = in_scores[in_index];
  }
}

int RPNProposalSSDPlugin::enqueue(int batchSize, const void *const *inputs,
                                  void **outputs, void *workspace,
                                  cudaStream_t stream) {
  // dimsNCHW: [N, 2 * num_anchor_per_point, H, W]
  const float *rpn_cls_prob_reshape = reinterpret_cast<const float*>(inputs[0]);
  // TODO(cjh): 4 * num_anchor_per_point or reverse?
  // dimsNCHW: [N, num_anchor_per_point * 4, H, W]
  const float *rpn_bbox_pred = reinterpret_cast<const float*>(inputs[1]);
  // dims: [N, 6, 1, 1] (axis-1: height, width, scale, origin_h, origin_w, 0)
  const float *im_info = reinterpret_cast<const float*>(inputs[2]);
  float *out_rois = reinterpret_cast<float*>(outputs[0]);

  float *host_im_info = new float[batchSize * 6]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_im_info, im_info,
                             batchSize * 6 * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
  std::ofstream f_im_info;
  std::string f_im_info_str = "/apollo/debug/rpn_im_info.txt";
  struct stat f_im_info_buffer;
  if (stat(f_im_info_str.c_str(), &f_im_info_buffer) != 0) {
    f_im_info.open(f_im_info_str.c_str());
    for (int i = 0; i < batchSize * 6; ++i) {
      if (i % 6 == 0) {
        f_im_info << "\n";
      }
      f_im_info << host_im_info[i] << " ";
    }
  }
#endif

  const int origin_height = (int)(host_im_info[0]);
  const int origin_width = (int)(host_im_info[1]);
  int num_anchor = height_ * width_ * num_anchor_per_point_;
  int rpn_bbox_pred_size = batchSize * num_anchor * 4;
  int scores_size = batchSize * num_anchor * 2;
  int anchors_size = num_anchor * 4;
  int out_rois_size = batchSize * top_n_ * 5;

  thrust::fill(thrust::device, out_rois, out_rois + size_t(out_rois_size), -1.0f);
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
  float *host_out_rois = new float[out_rois_size]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_out_rois, out_rois,
                                  out_rois_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::ofstream f_out_rois;
  std::string f_out_rois_str = "/apollo/debug/rcnn_init_out_rois.txt";
  struct stat out_rois_buffer;
  if (stat(f_out_rois_str.c_str(), &out_rois_buffer) != 0) {
    f_out_rois.open(f_out_rois_str.c_str());
    for (int i = 0; i < out_rois_size; ++i) {
      if (i % 5 == 0) {
        f_out_rois << "\n";
      }
      f_out_rois << host_out_rois[i] << " ";
    }
  }
  delete[] host_out_rois;
#endif

#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
  float *host_rpn_cls_prob_reshape = new float[scores_size]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_rpn_cls_prob_reshape, rpn_cls_prob_reshape,
                                  scores_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::ofstream f_rpn_cls_prob_reshape;
  std::string f_rpn_cls_prob_reshape_str = "/apollo/debug/rpn_cls_prob_reshape.txt";
  struct stat rpn_cls_prob_reshape_buffer;
  if (stat(f_rpn_cls_prob_reshape_str.c_str(), &rpn_cls_prob_reshape_buffer) != 0) {
    f_rpn_cls_prob_reshape.open(f_rpn_cls_prob_reshape_str.c_str());
    for (int i = 0; i < scores_size; ++i) {
      if (i % 10 == 0) {
        f_rpn_cls_prob_reshape << "\n";
      }
      f_rpn_cls_prob_reshape << host_rpn_cls_prob_reshape[i] << " ";
    }
  }
  delete[] host_rpn_cls_prob_reshape;
#endif
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
  float *host_rpn_bbox_pred = new float[rpn_bbox_pred_size]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_rpn_bbox_pred, rpn_bbox_pred,
                                  rpn_bbox_pred_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::ofstream f_rpn_bbox_pred;
  std::string f_rpn_bbox_pred_str = "/apollo/debug/rpn_bbox_pred.txt";
  struct stat f_rpn_bbox_pred_buffer;
  if (stat(f_rpn_bbox_pred_str.c_str(), &f_rpn_bbox_pred_buffer) != 0) {
    f_rpn_bbox_pred.open(f_rpn_bbox_pred_str.c_str());
    for (int i = 0; i < rpn_bbox_pred_size; ++i) {
      if (i % 4 == 0) {
        f_rpn_bbox_pred << "\n";
      }
      f_rpn_bbox_pred << host_rpn_bbox_pred[i] << " ";
    }
  }
  delete[] host_rpn_bbox_pred;
#endif

  int block_size, nthreads;

  // reshape to [N, num_anchor, 4]
  float *temp_rpn_bbox_pred;
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&temp_rpn_bbox_pred),
                             rpn_bbox_pred_size * sizeof(float)));
  nthreads = rpn_bbox_pred_size;
  block_size = (nthreads - 1) / thread_size_ + 1;
  reshape_boxes_kernel<<<block_size, thread_size_, 0, stream>>>(nthreads,
      rpn_bbox_pred, height_, width_, num_anchor_per_point_, temp_rpn_bbox_pred);
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
#endif

#if DEBUG
//  float *debug_rpn_bbox_pred;
//  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&debug_rpn_bbox_pred), rpn_bbox_pred_size * sizeof(float)));
//  repeatedly_mul_kernel<<<block_size, thread_size_, 0, stream>>>(nthreads,
//      temp_rpn_bbox_pred, debug_rpn_bbox_pred, bbox_std_, 4);
//  BASE_CUDA_CHECK(cudaDeviceSynchronize());
#endif

  // Normalization
  float *dev_bbox_mean, *dev_bbox_std;
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_bbox_mean), 4 * sizeof(float)));
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_bbox_std), 4 * sizeof(float)));
  BASE_CUDA_CHECK(cudaMemcpyAsync(dev_bbox_mean, bbox_mean_, 4 * sizeof(float), cudaMemcpyHostToDevice, stream));
  BASE_CUDA_CHECK(cudaMemcpyAsync(dev_bbox_std, bbox_std_, 4 * sizeof(float), cudaMemcpyHostToDevice, stream));
  repeatedly_mul_cuda(block_size, thread_size_, 0, stream, nthreads,
      temp_rpn_bbox_pred, temp_rpn_bbox_pred, dev_bbox_std, 4);
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
#endif
  repeatedly_add_cuda(block_size, thread_size_, 0, stream, nthreads,
      temp_rpn_bbox_pred, temp_rpn_bbox_pred, dev_bbox_mean, 4);
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
#endif

  // generate anchors
  float *anchors, *dev_anchor_heights, *dev_anchor_widths;
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&anchors),
                             anchors_size * sizeof(float)));
  BASE_CUDA_CHECK(cudaMemsetAsync(anchors, 0, anchors_size * sizeof(float), stream));
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_anchor_heights),
                             num_anchor_per_point_ * sizeof(float)));
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_anchor_widths),
                             num_anchor_per_point_ * sizeof(float)));
  BASE_CUDA_CHECK(cudaMemsetAsync(dev_anchor_heights, 0, num_anchor_per_point_ * sizeof(float), stream));
  BASE_CUDA_CHECK(cudaMemsetAsync(dev_anchor_widths, 0, num_anchor_per_point_ * sizeof(float), stream));
  BASE_CUDA_CHECK(cudaMemcpyAsync(dev_anchor_heights, anchor_heights_,
                             num_anchor_per_point_ * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  BASE_CUDA_CHECK(cudaMemcpyAsync(dev_anchor_widths, anchor_widths_,
                             num_anchor_per_point_ * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  block_size = (anchors_size - 1) / thread_size_ + 1;
  generate_anchors_kernel<<<block_size, thread_size_, 0, stream>>>(height_,
      width_, heat_map_a_, num_anchor_per_point_, dev_anchor_heights,
      dev_anchor_widths, anchors);
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
#endif

  // decode bbox
  float *proposals;
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&proposals),
                             rpn_bbox_pred_size * sizeof(float)));
  BASE_CUDA_CHECK(cudaMemsetAsync(proposals, 0, rpn_bbox_pred_size * sizeof(float), stream));
  nthreads = batchSize * num_anchor;
  block_size = (nthreads - 1) / thread_size_ + 1;
  bbox_transform_inv_cuda(block_size, thread_size_, 0, stream, nthreads,
      anchors, temp_rpn_bbox_pred, num_anchor, 1, proposals);
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
  float *host_proposals = new float[rpn_bbox_pred_size]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_proposals, proposals,
      rpn_bbox_pred_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::ofstream f_proposals;
  std::string f_proposals_str = "/apollo/debug/proposals.txt";
  struct stat buffer;
  if (stat(f_proposals_str.c_str(), &buffer) != 0) {
    f_proposals.open(f_proposals_str.c_str());
    for (int i = 0; i < rpn_bbox_pred_size; ++i) {
      if (i % 4 == 0) {
        f_proposals << "\n";
      }
      f_proposals << host_proposals[i] << " ";
    }
  }
  delete[] host_proposals;
#endif

  // clip boxes, i.e. refine proposals which are out of map
  if (refine_out_of_map_bbox_) {
    nthreads = rpn_bbox_pred_size;
    block_size = (nthreads - 1) / thread_size_ + 1;
    clip_boxes_cuda(block_size, thread_size_, 0, stream, nthreads,
        proposals, (float)origin_height, (float)origin_width);
  }
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
#endif

  // reshape scores to [N, num_anchor, 2]
  float *temp_scores;
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&temp_scores),
                             scores_size * sizeof(float)));
  nthreads = scores_size;
  block_size = (nthreads - 1) / thread_size_ + 1;
  reshape_scores_kernel<<<block_size, thread_size_, 0, stream>>>(nthreads,
      rpn_cls_prob_reshape, height_, width_, num_anchor_per_point_, temp_scores);
#if DEBUG
//  std::fstream fout;
//  fout.open("/apollo/debug/reshaped_scores.txt");
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
//  float *host_scores = new float[scores_size]();
//  fout << "initial scores at 2: \n" << host_scores[2] << std::endl;
//  BASE_CUDA_CHECK(cudaMemcpy(host_scores, temp_scores, scores_size * sizeof(float), cudaMemcpyDeviceToHost));
//  fout << "reshaped scores: \n" << host_scores;
//  delete[] host_scores;
#endif

  // filter boxes according to min_size_mode and threshold_objectness
  float *filtered_proposals, *filtered_scores;
  int *filtered_count;
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&filtered_proposals),
                             rpn_bbox_pred_size * sizeof(float)));
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&filtered_scores),
                             batchSize * num_anchor * sizeof(float)));
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&filtered_count),
                             batchSize * sizeof(int)));
  BASE_CUDA_CHECK(cudaMemsetAsync(filtered_proposals, 0,
                             rpn_bbox_pred_size * sizeof(float), stream));
  BASE_CUDA_CHECK(cudaMemsetAsync(filtered_scores, 0,
                             batchSize * num_anchor * sizeof(float), stream));
  BASE_CUDA_CHECK(cudaMemsetAsync(filtered_count, 0, batchSize * sizeof(int), stream));
  nthreads = batchSize * num_anchor;
  block_size = (nthreads - 1) / thread_size_ + 1;
  // TODO(cjh): filter area
  filter_boxes_cuda(block_size, thread_size_, 0, stream, nthreads,
      proposals, temp_scores, num_anchor, 1, 2, 0, 1, min_size_mode_,
      min_size_h_, min_size_w_, threshold_objectness_, filtered_proposals,
      filtered_scores, filtered_count);
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
#endif

  int *host_filtered_count = new int[batchSize]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_filtered_count, filtered_count,
                             batchSize * sizeof(int), cudaMemcpyDeviceToHost, stream));
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
#endif

  // descending sort proposals by score
  int *sorted_indexes;
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sorted_indexes),
                             batchSize * num_anchor * sizeof(int)));
  for (int i = 0; i < batchSize; ++i) {
    thrust::sequence(thrust::device, sorted_indexes + i * num_anchor,
                     sorted_indexes + i * num_anchor + host_filtered_count[i]);
    thrust::sort_by_key(thrust::device, filtered_scores + size_t(i * num_anchor),
                        filtered_scores + size_t(i * num_anchor + host_filtered_count[i]),
                        sorted_indexes + i * num_anchor, thrust::greater<float>());
  }
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
#endif

  // keep max N candidates
  float *pre_nms_proposals;
  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pre_nms_proposals),
                             batchSize * max_candidate_n_ * 4 * sizeof(float)));
  BASE_CUDA_CHECK(cudaMemsetAsync(pre_nms_proposals, 0,
                             batchSize * max_candidate_n_ * 4 * sizeof(float), stream));
  nthreads = batchSize * max_candidate_n_;
  block_size = (nthreads - 1) / thread_size_ + 1;
  keep_topN_boxes_cuda(block_size, thread_size_, 0, stream, nthreads,
      filtered_proposals, nullptr, sorted_indexes, filtered_count, false, num_anchor,
      max_candidate_n_, pre_nms_proposals, nullptr);
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
#endif

  // Nms, keep top N proposals and output final proposals
  // output dims: [num_roi, 5] (axis-1: batch_id, x_min, y_min, x_max, y_max)
  // TODO(chenjiahao): batch parallel nms
  int acc_box_num = 0;
  for (int i = 0; i < batchSize; ++i) {
    int cur_filter_count = std::min(host_filtered_count[i], max_candidate_n_);
    NmsForward(false, cur_filter_count, 4, overlap_ratio_, max_candidate_n_, top_n_, i, 0,
        pre_nms_proposals + size_t(i * max_candidate_n_ * 4), nullptr,
        out_rois + size_t(acc_box_num * 5),
        &acc_box_num, stream);
  }
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
#endif

  out_rois_num_ = acc_box_num;

  // Free cuda memory
  BASE_CUDA_CHECK(cudaFree(temp_rpn_bbox_pred));
  BASE_CUDA_CHECK(cudaFree(dev_bbox_mean));
  BASE_CUDA_CHECK(cudaFree(dev_bbox_std));
  BASE_CUDA_CHECK(cudaFree(anchors));
  BASE_CUDA_CHECK(cudaFree(dev_anchor_heights));
  BASE_CUDA_CHECK(cudaFree(dev_anchor_widths));
  BASE_CUDA_CHECK(cudaFree(proposals));
  BASE_CUDA_CHECK(cudaFree(temp_scores));
  BASE_CUDA_CHECK(cudaFree(filtered_proposals));
  BASE_CUDA_CHECK(cudaFree(filtered_scores));
  BASE_CUDA_CHECK(cudaFree(filtered_count));
  BASE_CUDA_CHECK(cudaFree(sorted_indexes));
  BASE_CUDA_CHECK(cudaFree(pre_nms_proposals));

  // Free host memory
  delete[] host_im_info;
  delete[] host_filtered_count;

  return 0;
}
}  // namespace inference
}  // namespace perception
}  // namespace apollo