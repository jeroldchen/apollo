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

/*
 * Just for debugging.
 */

#include "modules/perception/inference/tensorrt/plugins/rcnn_proposal_plugin.h"

#include <algorithm>
#include <iostream>
#include <vector>

#define DEBUG 1

#if DEBUG
#include <fstream>
#include <sys/stat.h>
#endif

namespace apollo {
namespace perception {
namespace inference {

using std::vector;

template<typename DType>
struct BBox {
  int id;
  DType score;
  DType x1;
  DType y1;
  DType x2;
  DType y2;
  std::vector<DType> prbs;

  static bool greater(const BBox<DType>& bbox1, const BBox<DType>& bbox2) {
    return bbox1.score >= bbox2.score;
  }
};

template <typename Dtype>
void targets2coords(const Dtype tg0, const Dtype tg1, const Dtype tg2, const Dtype tg3,
                    const Dtype acx, const Dtype acy, const Dtype acw, const Dtype ach,
                    const bool use_target_type_rcnn, const bool do_bbox_norm,
//                    const vector<Dtype>& bbox_means, const vector<Dtype>& bbox_stds,
                    const Dtype *bbox_means, const Dtype *bbox_stds,
                    Dtype& ltx, Dtype& lty, Dtype& rbx, Dtype& rby, bool bbox_size_add_one) {
  Dtype ntg0 = tg0, ntg1 = tg1, ntg2 = tg2, ntg3 = tg3;
  if (do_bbox_norm) {
    ntg0 *= bbox_stds[0]; ntg0 += bbox_means[0];
    ntg1 *= bbox_stds[1]; ntg1 += bbox_means[1];
    ntg2 *= bbox_stds[2]; ntg2 += bbox_means[2];
    ntg3 *= bbox_stds[3]; ntg3 += bbox_means[3];
  }
  if (use_target_type_rcnn) {
    Dtype bsz01 = bbox_size_add_one ? Dtype(1.0) : Dtype(0.0);
    Dtype ctx = ntg0 * acw + acx;
    Dtype cty = ntg1 * ach + acy;
    Dtype tw = Dtype(acw * std::exp(ntg2));
    Dtype th = Dtype(ach * std::exp(ntg3));
    ltx = Dtype(ctx - 0.5 * (tw - bsz01));
    lty = Dtype(cty - 0.5 * (th - bsz01));
    rbx = Dtype(ltx + tw - bsz01);
    rby = Dtype(lty + th - bsz01);
  } else {
    ltx = ntg0 + acx;
    lty = ntg1 + acy;
    rbx = ntg2 + acx;
    rby = ntg3 + acy;
  }
}

template <typename Dtype>
vector<bool> nms(vector< BBox<Dtype> >& candidates, const Dtype overlap, const int top_N, const bool addScore, const int max_candidate_N, bool bbox_size_add_one, bool voting, Dtype vote_iou)
{
  if (!voting && overlap >= Dtype(1.0)) {
    return vector<bool>(candidates.size(), true);
  }

  Dtype bsz01 = bbox_size_add_one ? Dtype(1.0) : Dtype(0.0);
  //Timer tm;
  //tm.Start();
  std::stable_sort(candidates.begin(), candidates.end(), BBox<Dtype>::greater);
  //LOG(INFO)<<"nms sort time: "<<tm.MilliSeconds();
  vector<bool> mask(candidates.size(), false);
  if (mask.size() == 0)
    return mask;
  int consider_size = candidates.size();
  if(max_candidate_N > 0) {
    consider_size = std::min<int>(consider_size, max_candidate_N);
  }
  vector<bool> skip(consider_size, false);
  vector<float> areas(consider_size, 0);
  for (int i = 0; i < consider_size; ++i)
  {
    areas[i] = (candidates[i].x2 - candidates[i].x1 + bsz01) * (candidates[i].y2- candidates[i].y1 + bsz01);
  }
  for (int count = 0, i = 0; count < top_N && i < consider_size; ++i)
  {
    if (skip[i])
      continue;
    mask[i] = true;
    ++count;
    Dtype s_vt = candidates[i].score;
    Dtype x1_vt = 0.0;
    Dtype y1_vt = 0.0;
    Dtype x2_vt = 0.0;
    Dtype y2_vt = 0.0;
    if (voting) {
      CHECK_GE(s_vt, 0);
      x1_vt = candidates[i].x1 * s_vt;
      y1_vt = candidates[i].y1 * s_vt;
      x2_vt = candidates[i].x2 * s_vt;
      y2_vt = candidates[i].y2 * s_vt;
    }
    // suppress the significantly covered bbox
    for (int j = i + 1; j < consider_size; ++j)
    {
      if (skip[j]) continue;
      // get intersections
      float xx1 = MAX(candidates[i].x1, candidates[j].x1);
      float yy1 = MAX(candidates[i].y1, candidates[j].y1);
      float xx2 = MIN(candidates[i].x2, candidates[j].x2);
      float yy2 = MIN(candidates[i].y2, candidates[j].y2);
      float w = xx2 - xx1 + bsz01;
      float h = yy2 - yy1 + bsz01;
      if (w > 0 && h > 0)
      {
        // compute overlap
        //float o = w * h / areas[j];
        float o = w * h;
        o = o / (areas[i] + areas[j] - o);
        if (voting && o > vote_iou) {
          Dtype s_vt_cur = candidates[j].score;
          CHECK_GE(s_vt_cur, 0);
          s_vt += s_vt_cur;
          x1_vt += candidates[j].x1 * s_vt_cur;
          y1_vt += candidates[j].y1 * s_vt_cur;
          x2_vt += candidates[j].x2 * s_vt_cur;
          y2_vt += candidates[j].y2 * s_vt_cur;
        }
      }
    }
    if (voting && s_vt > 0.0001) {
      candidates[i].x1 = x1_vt / s_vt;
      candidates[i].y1 = y1_vt / s_vt;
      candidates[i].x2 = x2_vt / s_vt;
      candidates[i].y2 = y2_vt / s_vt;
    }
  }
  return mask;
}

int RCNNProposalPlugin::enqueue(int batchSize, const void *const *inputs,
                                  void **outputs, void *workspace,
                                  cudaStream_t stream) {
  // cls_score_softmax dims: [num_rois, 4, 1, 1]
  const float* dev_cls_score_softmax = reinterpret_cast<const float*>(inputs[0]);
  // bbox_pred dims: [num_rois, 4 * 4 (box_dim * num_class), 1, 1]
  const float* dev_bbox_pred = reinterpret_cast<const float*>(inputs[1]);
  // rois dims: [num_rois, 5, 1, 1]
  const float* dev_rois = reinterpret_cast<const float*>(inputs[2]);
  // im_info dims: [N, 6, 1, 1]
  const float* dev_im_info = reinterpret_cast<const float*>(inputs[3]);

  int cls_score_softmax_size = num_rois_ * 4;
  int bbox_pred_size = num_rois_ * 4 * 4;
  int output_size = batchSize * top_n_ * out_channel_;

  float *host_cls_score_softmax = new float[cls_score_softmax_size]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_cls_score_softmax, dev_cls_score_softmax,
                                  cls_score_softmax_size * sizeof(float), cudaMemcpyDeviceToHost, stream));

  float *host_bbox_pred = new float[bbox_pred_size]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_bbox_pred, dev_bbox_pred,
                                  bbox_pred_size * sizeof(float), cudaMemcpyDeviceToHost, stream));

  float *host_im_info = new float[batchSize * 6]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_im_info, dev_im_info,
                                  batchSize * 6 * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));

  float *host_rois = new float[num_rois_ * 5]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_rois, dev_rois,
                                  num_rois_ * 5 * sizeof(float), cudaMemcpyDeviceToHost, stream));

  bool has_img_info_ = true;
  bool bbox_size_add_one_ = true;
  int num_rpns_ = 1;
  bool use_target_type_rcnn_ = true;
  bool do_bbox_norm_ = true;
  int rois_dim_ = 5;
  bool force_retain_all_ = false;

  float im_height, im_width;
  vector<float> input_width = vector<float>(1, 0);
  vector<float> input_height = vector<float>(1, 0);
  vector<float> im_width_scale = vector<float>(1, 0);
  vector<float> im_height_scale = vector<float>(1, 0);
  vector<float> cords_offset_x = vector<float>(1, float(0));
  vector<float> cords_offset_y = vector<float>(1, 0);
  float min_size_w_cur = this->min_size_w_;
  float min_size_h_cur = this->min_size_h_;
  if (has_img_info_) {
//    CHECK_EQ(bottom.back()->count(1), 6);
    const float* img_info_data = host_im_info;
    im_width = img_info_data[0];
    im_height = img_info_data[1];
    input_width.clear();
    input_height.clear();
    im_width_scale.clear();
    im_height_scale.clear();
    cords_offset_x.clear();
    cords_offset_y.clear();
    for (int n = 0; n < batchSize; n++) {
      input_width.push_back(img_info_data[n * 6 + 0]);
      input_height.push_back(img_info_data[n * 6 + 1]);
      CHECK_GT(input_width[n], 0);
      CHECK_GT(input_height[n], 0);
      im_width_scale.push_back(img_info_data[n * 6 + 2]);
      im_height_scale.push_back(img_info_data[n * 6 + 3]);
      CHECK_GT(im_width_scale[n], 0);
      CHECK_GT(im_height_scale[n], 0);
      cords_offset_x.push_back(img_info_data[n * 6 + 4]);
      cords_offset_y.push_back(img_info_data[n * 6 + 5]);
    }
  }

  float bsz01 = bbox_size_add_one_ ? float(1.0) : float(0.0);

  const int num_rois = num_rois_;
  const int probs_dim = 4;
  const int cords_dim = 16;
  const int pre_rois_dim = 5;
  CHECK_EQ(probs_dim, this->num_class_ + 1);
  if (this->regress_agnostic_) {
    CHECK_EQ(cords_dim, 2 * 4);
  } else {
    CHECK_EQ(cords_dim, (this->num_class_ + 1) * 4);
  }
  CHECK_EQ(pre_rois_dim, 5); // imid, x1, y1, x2, y2

//  const float* probs_st = bottom[0]->cpu_data();
//  const float* cords_st = bottom[1]->cpu_data();
//  const float* rois_st = bottom[2]->cpu_data();

  vector<vector<BBox<float> > > proposal_per_img_vec;
  for (int i = 0; i < num_rois; ++i) {
//    const float* probs = probs_st + i * probs_dim;
//    const float* cords = cords_st + i * cords_dim;
//    const float* rois = rois_st + i * pre_rois_dim;
    // filter those width low probs
    if ((1.0 - host_cls_score_softmax[i * 4]) < this->threshold_objectness_) {
      if (!force_retain_all_) {
        continue;
      }
    }
    float score_max = -100;
    int cls_max = -1;
    for (int c = 0; c < this->num_class_; c++) {
      float score_c = host_cls_score_softmax[i * 4 + c + 1] - this->thresholds_[c];
      if (score_c > score_max) {
        score_max = score_c;
        cls_max = c;
      }
    }
    if (score_max < 0) {
      if (!force_retain_all_) {
        continue;
      }
    }
    CHECK_GE(cls_max, 0);

    int imid = int(host_rois[i * 5]);
    if (imid == -1) {
      continue;
    }

    int cdst = this->regress_agnostic_ ? 4 : (cls_max + 1) * 4;

    BBox<float> bbox;
    bbox.id = imid;
//    bbox.roi_id = i;
    bbox.score = host_cls_score_softmax[i * 4 + cls_max + 1];
    float ltx, lty, rbx, rby;
    float rois_w = host_rois[i * 5 + 3] - host_rois[i * 5 + 1] + bsz01;
    float rois_h = host_rois[i * 5 + 4] - host_rois[i * 5 + 2] + bsz01;
    float rois_ctr_x = host_rois[i * 5 + 1] + 0.5 * (rois_w - bsz01);
    float rois_ctr_y = host_rois[i * 5 + 2] + 0.5 * (rois_h - bsz01);

    float input_width_cur = input_width.size() > 1 ? input_width[imid] : input_width[0];
    float input_height_cur = input_height.size() > 1 ? input_height[imid] : input_height[0];

      targets2coords<float>(host_bbox_pred[i * 16 + cdst], host_bbox_pred[i * 16 + cdst + 1],
                            host_bbox_pred[i * 16 + cdst + 2], host_bbox_pred[i * 16 + cdst + 3],
                            rois_ctr_x, rois_ctr_y, rois_w, rois_h,
                            use_target_type_rcnn_, do_bbox_norm_,
                            bbox_mean_, bbox_std_,
                            ltx, lty, rbx, rby, bbox_size_add_one_);

    bbox.x1 = ltx;
    bbox.y1 = lty;
    bbox.x2 = rbx;
    bbox.y2 = rby;

    if (this->refine_out_of_map_bbox_) {
      bbox.x1 = MIN(MAX(bbox.x1, 0), input_width_cur-1);
      bbox.y1 = MIN(MAX(bbox.y1, 0), input_height_cur-1);
      bbox.x2 = MIN(MAX(bbox.x2, 0), input_width_cur-1);
      bbox.y2 = MIN(MAX(bbox.y2, 0), input_height_cur-1);
    }

    float bw = bbox.x2 - bbox.x1 + bsz01;
    float bh = bbox.y2 - bbox.y1 + bsz01;
    if (this->min_size_mode_ == 0) {
      if (bw < min_size_w_cur || bh < min_size_h_cur) {
        if (!force_retain_all_) {
          continue;
        }
      }
    } else if (this->min_size_mode_ == 1) {
      if (bw < min_size_w_cur && bh < min_size_h_cur) {
        if (!force_retain_all_) {
          continue;
        }
      }
    } else {
      CHECK(false);
    }

      if (imid + 1 > proposal_per_img_vec.size()) {
        proposal_per_img_vec.resize(imid + 1);
      }
      for (int c = 0; c < this->num_class_ + 1; ++c) {
        bbox.prbs.push_back(host_cls_score_softmax[i * 4 + c]);
      }
      proposal_per_img_vec[imid].push_back(bbox);

  }

    vector<vector<BBox<float> > > proposal_batch_vec(1);
    for (int i = 0; i < proposal_per_img_vec.size(); ++i) {
      vector<BBox<float> >& proposal_cur = proposal_per_img_vec[i];
      //do nms
      vector<bool> sel;
      if (force_retain_all_) {
        sel.resize(proposal_cur.size(), true);
      } else {
          sel = nms<float>(proposal_cur, overlap_ratio_, top_n_,
                           false, max_candidate_n_, bbox_size_add_one_,
                           false, 0.7);
      }
      for(int k = 0; k < sel.size(); k++) {
        if(sel[k]) {
          float bw = proposal_cur[k].x2 - proposal_cur[k].x1 + bsz01;
          float bh = proposal_cur[k].y2 - proposal_cur[k].y1 + bsz01;
          if(bw <= 0 || bh <= 0) {
            if (!force_retain_all_) {
              continue;
            }
          }
          float bwxh = bw * bh;
          for(int t = 0; t < 1; t++) {
              proposal_batch_vec[t].push_back(proposal_cur[k]);
          }
        }
      }
    }
    for(int t = 0; t < 1; t++) {
        const int top_num = proposal_batch_vec[t].size();
      float* top_boxes_scores = new float[top_num];
      for (int k = 0; k < top_num; k++) {
          top_boxes_scores[k*rois_dim_] = proposal_batch_vec[t][k].id;
          top_boxes_scores[k*rois_dim_+1] = proposal_batch_vec[t][k].x1;
          top_boxes_scores[k*rois_dim_+2] = proposal_batch_vec[t][k].y1;
          top_boxes_scores[k*rois_dim_+3] = proposal_batch_vec[t][k].x2;
          top_boxes_scores[k*rois_dim_+4] = proposal_batch_vec[t][k].y2;
          if (this->rpn_proposal_output_score_) {
            for (int c = 0; c < this->num_class_ + 1; c++) {
              top_boxes_scores[k * rois_dim_ + 5 + c] = proposal_batch_vec[t][k].prbs[c];
            }
          }
        }
#if DEBUG
      std::ofstream f_top;
      std::string f_top_str = "/apollo/debug_cpu/rcnn_bboxes.txt";
      f_top.open(f_top_str.c_str());
      for (int i = 0; i < top_num * 9; ++i) {
        if (i != 0 && i % 9 == 0) {
          f_top << std::endl;
        }
        f_top << top_boxes_scores[i] << " ";
      }
      f_top.close();
#endif
    }

  return 0;

}
}  // namespace inference
}  // namespace perception
}  // namespace apollo