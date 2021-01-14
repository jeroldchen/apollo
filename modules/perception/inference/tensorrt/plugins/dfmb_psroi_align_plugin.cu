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

/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "modules/perception/inference/tensorrt/plugins/dfmb_psroi_align_plugin.h"

#include "modules/perception/inference/tensorrt/plugins/kernels.h"

#define DEBUG 1

#if DEBUG
#include <fstream>
#include <sys/stat.h>
#endif

namespace apollo {
namespace perception {
namespace inference {

template <typename Dtype>
__global__ void DFMBPSROIAlignForward(
    const int nthreads, const Dtype* bottom_data,
    const Dtype heat_map_a, const Dtype heat_map_b,
    const Dtype pad_ratio, const int batch_size, const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, const Dtype* bottom_trans,
    const bool no_trans, const Dtype trans_std,
    const int sample_per_part, const int output_channel,
    const int group_height, const int group_width,
    const int part_height, const int part_width,
    const int num_classes, const int channels_each_class,
    Dtype* top_data) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_channel;
    int n = index / pooled_width / pooled_height / output_channel;

    // [start, end) interval for spatial sampling
    const Dtype* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    if (roi_batch_ind < 0 || roi_batch_ind >= batch_size) {
      // TODO(cjh): debug
//      top_data[index] = (Dtype)0;
      continue;
    }

    Dtype pad_w = (offset_bottom_rois[3] - offset_bottom_rois[1] + 1) * pad_ratio;
    Dtype pad_h = (offset_bottom_rois[4] - offset_bottom_rois[2] + 1) * pad_ratio;
    Dtype roi_start_w = (offset_bottom_rois[1] - pad_w - heat_map_b) / heat_map_a;
    Dtype roi_start_h = (offset_bottom_rois[2] - pad_h - heat_map_b) / heat_map_a;
    Dtype roi_end_w = (offset_bottom_rois[3] + pad_w - heat_map_b) / heat_map_a;
    Dtype roi_end_h = (offset_bottom_rois[4] + pad_h - heat_map_b) / heat_map_a;
    // Force too small ROIs to be 1x1
    Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);
    Dtype roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0

    // Compute w and h at bottom
    Dtype bin_size_h = roi_height / (Dtype)pooled_height;
    Dtype bin_size_w = roi_width / (Dtype)pooled_width;

    Dtype sub_bin_size_h = bin_size_h / (Dtype)sample_per_part;
    Dtype sub_bin_size_w = bin_size_w / (Dtype)sample_per_part;

    int part_h = floor((Dtype)ph) / pooled_height * part_height;
    int part_w = floor((Dtype)pw) / pooled_width * part_width;
    int class_id = ctop / channels_each_class;
    Dtype trans_x = no_trans ? (Dtype)0 :
                    bottom_trans[(((n * num_classes + class_id) * 2) *
                        part_height + part_h) * part_width + part_w] * trans_std;
    Dtype trans_y = no_trans ? (Dtype)0 :
                    bottom_trans[(((n * num_classes + class_id) * 2 + 1) *
                        part_height + part_h) * part_width + part_w] * trans_std;

    int hstart = (Dtype)ph * bin_size_h +
        roi_start_h + trans_y * roi_height;
    int wstart =  (Dtype)pw* bin_size_w +
        roi_start_w + trans_x * roi_width;

    Dtype sum = 0;
    int count = 0;
    int gh = floor((Dtype)ph* group_height / pooled_height);
    int gw = floor((Dtype)pw * group_width / pooled_width);
    gh = min(max(gh, 0), group_height - 1);
    gw = min(max(gw, 0), group_width - 1);

    const Dtype* offset_bottom_data = bottom_data +
        (roi_batch_ind * channels) * height * width;
    for (int ih = 0; ih < sample_per_part; ih++) {
      for (int iw = 0; iw < sample_per_part; iw++) {
        Dtype w = wstart + (iw + 0.5) * sub_bin_size_w;
        Dtype h = hstart + (ih + 0.5) * sub_bin_size_h;
        // bilinear interpolation
        if (w <= -1 || w >= width || h <= -1 || h >= height) {
          continue;
        }
        int c = (ctop * group_height + gh) * group_width + gw;
        int x1 = floor(w);
        int x2 = ceil(w);
        int y1 = floor(h);
        int y2 = ceil(h);
        Dtype dist_x = (Dtype)w - x1;
        Dtype dist_y = (Dtype)h - y1;
        const Dtype* data = offset_bottom_data + c * height * width;
        Dtype value11 = (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) ? data[y1 * width + x1] : Dtype(0.0);
        Dtype value12 = (x1 >= 0 && x1 < width && y2 >= 0 && y2 < height) ? data[y2 * width + x1] : Dtype(0.0);
        Dtype value21 = (x2 >= 0 && x2 < width && y1 >= 0 && y1 < height) ? data[y1 * width + x2] : Dtype(0.0);
        Dtype value22 = (x2 >= 0 && x2 < width && y2 >= 0 && y2 < height) ? data[y2 * width + x2] : Dtype(0.0);
        Dtype value = (1 - dist_x) * (1 - dist_y) * value11
            + (1 - dist_x) * dist_y * value12
            + dist_x * (1 - dist_y) * value21
            + dist_x * dist_y * value22;
        sum += value;
        count++;
      }
    }
    top_data[index] = count == 0 ? (Dtype)0 : sum / count;
  }
}

int DFMBPSROIAlignPlugin::enqueue(int batchSize, const void *const *inputs,
                                  void **outputs, void *workspace,
                                  cudaStream_t stream) {
  const float *bottom_data = reinterpret_cast<const float*>(inputs[0]);
  const float *bottom_rois = reinterpret_cast<const float*>(inputs[1]);
  const float *bottom_trans = no_trans_ ? nullptr :
                              reinterpret_cast<const float*>(inputs[2]);
  float *top_data = reinterpret_cast<float*>(outputs[0]);
  int channels_each_class = no_trans_ ? output_channel_ :
                            output_channel_ / num_classes_;

  BASE_CUDA_CHECK(cudaMemsetAsync(top_data, 0, output_size_ * sizeof(float), stream));
  BASE_CUDA_CHECK(cudaDeviceSynchronize());

  // TODO(cjh): debugging rcnn plugin, so to make sure previous value is the same.
#if DEBUG
//  int bottom_data_size = batchSize * channels_ * height_ * width_;
//  float *host_temp_bottom_data = new float[bottom_data_size];
//  std::ifstream fin_ft_add_left_right;
//  std::string fin_ft_add_left_right_str = "/apollo/debug_input/ft_add_left_right.txt";
//  fin_ft_add_left_right.open(fin_ft_add_left_right_str.c_str());
//  for (int i = 0; i < bottom_data_size; ++i) {
//    fin_ft_add_left_right >> host_temp_bottom_data[i];
//  }
//  float *temp_bottom_data;
//  BASE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&temp_bottom_data), bottom_data_size * sizeof(float)));
//  BASE_CUDA_CHECK(cudaMemcpyAsync(temp_bottom_data, host_temp_bottom_data,
//                                  bottom_data_size * sizeof(float), cudaMemcpyHostToDevice, stream));
//  bottom_data = temp_bottom_data;
//  delete[] host_temp_bottom_data;
#endif

  int block_size = (output_size_ - 1) / thread_size_ + 1;
  DFMBPSROIAlignForward<<<block_size, thread_size_, 0, stream>>>(output_size_,
      bottom_data, heat_map_a_, heat_map_b_, pad_ratio_, batchSize, channels_, height_,
      width_, pooled_height_, pooled_width_, bottom_rois, bottom_trans,
      no_trans_, trans_std_, sample_per_part_, output_channel_, group_height_,
      group_width_, part_height_, part_width_, num_classes_,
      channels_each_class, top_data);

#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
  float *host_ft_add_left_right = new float[batchSize * channels_ * height_ * width_]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_ft_add_left_right, bottom_data,
                                  batchSize * channels_ * height_ * width_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::ofstream f_ft_add_left_right;
  std::string f_ft_add_left_right_str = "/apollo/debug/ft_add_left_right.txt";
  struct stat ft_add_left_right_buffer;
  if (stat(f_ft_add_left_right_str.c_str(), &ft_add_left_right_buffer) != 0) {
    f_ft_add_left_right.open(f_ft_add_left_right_str.c_str());
    for (int i = 0; i < batchSize * channels_ * height_ * width_; ++i) {
      if (i % 10 == 0) {
        f_ft_add_left_right << "\n";
      }
      f_ft_add_left_right << host_ft_add_left_right[i] << " ";
    }
  }
  delete[] host_ft_add_left_right;
#endif
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
  float *host_rois = new float[batchSize * 300 * 5]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_rois, bottom_rois,
                                  batchSize * 300 * 5 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::ofstream f_rois;
  std::string f_rois_str = "/apollo/debug/rois.txt";
  struct stat rois_buffer;
  if (stat(f_rois_str.c_str(), &rois_buffer) != 0) {
    f_rois.open(f_rois_str.c_str());
    for (int i = 0; i < batchSize * 300 * 5; ++i) {
      if (i % 5 == 0) {
        f_rois << "\n";
      }
      f_rois << host_rois[i] << " ";
    }
  }
  delete[] host_rois;
#endif
#if DEBUG
  BASE_CUDA_CHECK(cudaDeviceSynchronize());
  float *host_psroi_rois = new float[output_size_]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_psroi_rois, top_data,
                                  output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::ofstream f_psroi_rois;
  std::string f_psroi_rois_str = "/apollo/debug/psroi_rois.txt";
  struct stat psroi_rois_buffer;
  if (stat(f_psroi_rois_str.c_str(), &psroi_rois_buffer) != 0) {
    f_psroi_rois.open(f_psroi_rois_str.c_str());
    for (int i = 0; i < output_size_; ++i) {
      if (i % 10 == 0) {
        f_psroi_rois << "\n";
      }
      f_psroi_rois << host_psroi_rois[i] << " ";
    }
  }
  delete[] host_psroi_rois;
#endif

#if DEBUG
//  BASE_CUDA_CHECK(cudaFree(temp_bottom_data));
#endif

  return 0;
}
}  // namespace inference
}  // namespace perception
}  // namespace apollo
