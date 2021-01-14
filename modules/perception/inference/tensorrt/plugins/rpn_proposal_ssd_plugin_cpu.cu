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

#include "modules/perception/inference/tensorrt/plugins/rpn_proposal_ssd_plugin.h"

#include <algorithm>
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

int RPNProposalSSDPlugin::enqueue(int batchSize, const void *const *inputs,
                                  void **outputs, void *workspace,
                                  cudaStream_t stream) {
  // TODO(cjh): [fg, bg] or [bg, fg]?
  // dimsNCHW: [N, 2 * num_anchor_per_point, H, W]
  const float *rpn_cls_prob_reshape = reinterpret_cast<const float*>(inputs[0]);
  // TODO(cjh): 4 * num_anchor_per_point or reverse?
  // dimsNCHW: [N, 4 * num_anchor_per_point, H, W]
  const float *rpn_bbox_pred = reinterpret_cast<const float*>(inputs[1]);
  // dims: [N, 6, 1, 1] (axis-1: height, width, scale, origin_h, origin_w, 0)
  const float *im_info = reinterpret_cast<const float*>(inputs[2]);
  float *out_rois = reinterpret_cast<float*>(outputs[0]);

  int num_anchor = height_ * width_ * num_anchor_per_point_;
  int rpn_bbox_pred_size = batchSize * num_anchor * 4;
  int scores_size = batchSize * num_anchor * 2;
  int anchors_size = num_anchor * 4;
  int out_rois_size = batchSize * top_n_ * 5;

  float *host_rpn_cls_prob_reshape = new float[scores_size];
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_rpn_cls_prob_reshape, rpn_cls_prob_reshape,
                                  scores_size * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));

  float *host_rpn_bbox_pred = new float[rpn_bbox_pred_size];
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_rpn_bbox_pred, rpn_bbox_pred,
                                  rpn_bbox_pred_size * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));

  float *host_im_info = new float[batchSize * 6]();
  BASE_CUDA_CHECK(cudaMemcpyAsync(host_im_info, im_info,
                             batchSize * 6 * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));

  bool has_img_info_ = true;
  bool bbox_size_add_one_ = true;
  int num_rpns_ = 1;
  bool use_target_type_rcnn_ = true;
  bool do_bbox_norm_ = true;
  int rois_dim_ = 5;

  float anchor_x1_vec_[15] = {-4.11649, -7.5, -13.3564, -8.73298, -15.5, -27.2128,
                              -17.966, -31.5, -54.9256, -36.4319, -63.5, -110.351,
                              -73.3639, -127.5, -221.202};
  float anchor_x2_vec_[15] = {4.11649, 7.5, 13.3564, 8.73298, 15.5, 27.2128,
                              17.966, 31.5, 54.9256, 36.4319, 63.5, 110.351,
                              73.3639, 127.5, 221.202};
  float anchor_y1_vec_[15] = {-13.3633, -7.5, -4.1188, -27.2267, -15.5, -8.7376,
                              -54.9534, -31.5, -17.9752, -110.407, -63.5, -36.4504,
                              -221.313, -127.5, -73.4008};
  float anchor_y2_vec_[15] = {13.3633, 7.5, 4.1188, 27.2267, 15.5, 8.7376,
                              54.9534, 31.5, 17.9752, 110.407, 63.5, 36.4504,
                              221.313, 127.5, 73.4008};

//  float im_height = this->im_height_, im_width = this->im_width_;
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
//  if (this->refine_out_of_map_bbox_) {
//    if (top.size() == 0) {
//      CHECK_GT(im_width, 0);
//      CHECK_GT(im_height, 0);
//    }
//  }

  float bsz01 = bbox_size_add_one_ ? float(1.0) : float(0.0);

//  const int num = bottom[0]->num();
  const int num = batchSize;
  vector<vector<BBox<float> > > proposal_batch_vec(1);
  for (int i = 0; i < num; ++i) {
    float input_width_cur = input_width.size() > 1 ? input_width[i] : input_width[0];
    float input_height_cur = input_height.size() > 1 ? input_height[i] : input_height[0];
    vector<BBox<float> > proposal_cur;
    for (int r = 0; r < num_rpns_; ++r) {
//      int prob_idx = 2 * r;
//      int tgt_idx = prob_idx + 1;

//      CHECK_EQ(bottom[prob_idx]->num(), num);
//      CHECK_EQ(bottom[tgt_idx]->num(), num);
//      CHECK_EQ(bottom[prob_idx]->channels(), num_anchors_ * 2);
//      CHECK_EQ(bottom[tgt_idx]->channels(), num_anchors_ * 4);
//      CHECK_EQ(bottom[prob_idx]->height(), bottom[tgt_idx]->height());
//      CHECK_EQ(bottom[prob_idx]->width(), bottom[tgt_idx]->width());

//      const int map_height = bottom[prob_idx]->height();
//      const int map_width  = bottom[prob_idx]->width();
      const int map_height = height_;
      const int map_width  = width_;
      const int map_size   = map_height * map_width;
//      const float heat_map_a = this->heat_map_a_vec_[r];
      const float heat_map_a = heat_map_a_;
//      const float heat_map_b = this->heat_map_b_vec_[r];
      const float heat_map_b = 0;

//      const float* prob_data = bottom[prob_idx]->cpu_data();
      const float* prob_data = host_rpn_cls_prob_reshape;
//      const float* tgt_data = bottom[tgt_idx]->cpu_data();
      const float* tgt_data = host_rpn_bbox_pred;
      for (int a = 0; a < num_anchor_per_point_; ++a) {
//        int score_channel  = num_anchors_ + a;
        int offset_channel = 4 * a;
//        const float* scores = prob_data + bottom[prob_idx]->offset(i, score_channel, 0, 0);
//        const float* dx1 =    tgt_data + bottom[tgt_idx]->offset(i, offset_channel + 0, 0, 0);
//        const float* dy1 =    tgt_data + bottom[tgt_idx]->offset(i, offset_channel + 1, 0, 0);
//        const float* dx2 =    tgt_data + bottom[tgt_idx]->offset(i, offset_channel + 2, 0, 0);
//        const float* dy2 =    tgt_data + bottom[tgt_idx]->offset(i, offset_channel + 3, 0, 0);
        float anchor_width = anchor_x2_vec_[a] - anchor_x1_vec_[a] + bsz01;
        float anchor_height = anchor_y2_vec_[a] - anchor_y1_vec_[a] + bsz01;
        float anchor_ctr_x = anchor_x1_vec_[a] + 0.5 * (anchor_width - bsz01);
        float anchor_ctr_y = anchor_y1_vec_[a] + 0.5 * (anchor_height - bsz01);
//        float anchor_width = anchor_widths_[a];
//        float anchor_height = anchor_heights_[a];
//        float anchor_ctr_x = 0;
//        float anchor_ctr_y = 0;

        for(int off = 0; off< map_size; ++off)
        {
          float score_cur = 0.0;
          int cur_id = a * map_size + off;
          score_cur = prob_data[num_anchor + cur_id];
          if(score_cur < this->threshold_objectness_) {
            continue;
          }

          int h = off / map_width;
          int w = off % map_width ;
          float input_ctr_x = w * heat_map_a + heat_map_b + anchor_ctr_x;
          float input_ctr_y = h * heat_map_a + heat_map_b + anchor_ctr_y;

//          if (this->allow_border_ >= float(0.0)
//              || this->allow_border_ratio_ >= float(0.0)) {
            float x1 = input_ctr_x - 0.5 * (anchor_width - bsz01);
            float y1 = input_ctr_y - 0.5 * (anchor_height - bsz01);
            float x2 = x1 + anchor_width - bsz01;
            float y2 = y1 + anchor_height - bsz01;
//            if (this->allow_border_ >= float(0.0) && (
//                x1 < -this->allow_border_ || y1 < -this->allow_border_
//                    || x2 > input_width_cur - 1 + this->allow_border_ ||
//                    y2 > input_height_cur - 1 + this->allow_border_ )) {
//              continue;
//            } else if (this->allow_border_ratio_ >= float(0.0)) {
//              float x11 = max<float>(0, x1);
//              float y11 = max<float>(0, y1);
//              float x22 = min<float>(input_width_cur - 1, x2);
//              float y22 = min<float>(input_height_cur - 1, y2);
//              if ((y22 - y11 + bsz01) * (x22 - x11 + bsz01)
//                  / ((y2 - y1 + bsz01) * (x2 - x1 + bsz01))
//                  < (1.0 - this->allow_border_ratio_)) {
//                continue;
//              }
//            }
//          }

          BBox<float> bbox;
          bbox.id = i;
          bbox.score = score_cur;
          float ltx, lty, rbx, rby;
          float cur_dx1 = tgt_data[a * 4 * map_size + off];
          float cur_dy1 = tgt_data[a * 4 * map_size + map_size + off];
          float cur_dx2 = tgt_data[a * 4 * map_size + 2 * map_size + off];
          float cur_dy2 = tgt_data[a * 4 * map_size + 3 * map_size + off];
//          targets2coords<float>(dx1[off], dy1[off], dx2[off], dy2[off],
          targets2coords<float>(cur_dx1, cur_dy1, cur_dx2, cur_dy2,
                                input_ctr_x, input_ctr_y, anchor_width, anchor_height,
                                use_target_type_rcnn_, do_bbox_norm_,
                                this->bbox_mean_, this->bbox_std_, ltx, lty, rbx, rby, bbox_size_add_one_);
          bbox.x1 = ltx;
          bbox.y1 = lty;
          bbox.x2 = rbx;
          bbox.y2 = rby;
          if (this->refine_out_of_map_bbox_) {
            bbox.x1 = std::min(std::max(bbox.x1, 0.f), input_width_cur-1);
            bbox.y1 = std::min(std::max(bbox.y1, 0.f), input_height_cur-1);
            bbox.x2 = std::min(std::max(bbox.x2, 0.f), input_width_cur-1);
            bbox.y2 = std::min(std::max(bbox.y2, 0.f), input_height_cur-1);
          }

          float bw = bbox.x2 - bbox.x1 + bsz01;
          float bh = bbox.y2 - bbox.y1 + bsz01;
//          if (this->min_size_mode_ == DetectionOutputSSDParameter_MIN_SIZE_MODE_HEIGHT_AND_WIDTH) {
//            if (bw < min_size_w_cur || bh < min_size_h_cur) {
//              continue;
//            }
//          } else if (this->min_size_mode_ == DetectionOutputSSDParameter_MIN_SIZE_MODE_HEIGHT_OR_WIDTH) {
            if (bw < min_size_w_cur && bh < min_size_h_cur) {
              continue;
            }
//          } else {
//            CHECK(false);
//          }

//          if (top.size() != 0) {
            proposal_cur.push_back(bbox);
//          } else {
//            bbox.id = 0;
//            for (int c = 0; c < this->num_class_; ++c) {
//              this->all_candidate_bboxes_[c].push_back(bbox);
//            }
//          }
        }
      }
    }
#if DEBUG
    std::ofstream f_proposal_cur;
    std::string f_proposal_cur_str = "/apollo/debug_cpu/proposal_cur.txt";
    f_proposal_cur.open(f_proposal_cur_str.c_str());
    for (int i = 0; i < proposal_cur.size(); ++i) {
      if (i != 0 && i % 4 == 0) {
        f_proposal_cur << std::endl;
      }
      f_proposal_cur << proposal_cur[i].x1 << " " << proposal_cur[i].y1 << " "
                     << proposal_cur[i].x2 << " " << proposal_cur[i].y2 << " ";
    }
    f_proposal_cur.close();
#endif

//    if (top.size() != 0) {
      //do nms
      vector<bool> sel;
//      if(this->nms_use_soft_nms_[0]) {
        //Timer tm;
        //tm.Start();
//        sel = caffe::soft_nms(proposal_cur, this->nms_overlap_ratio_[0],
//                              this->nms_top_n_[0], this->nms_max_candidate_n_[0], bbox_size_add_one_,
//                              this->nms_voting_[0], this->nms_vote_iou_[0]);
        //LOG(INFO)<<"soft-nms time: "<<tm.MilliSeconds();
//      } else {
        //Timer tm;
        //tm.Start();
        sel = nms<float>(proposal_cur, this->overlap_ratio_, this->top_n_,
                         false, this->max_candidate_n_, bbox_size_add_one_, false, 0.7);
//                         this->nms_voting_[0], this->nms_vote_iou_[0]);
        //LOG(INFO)<<"nms time: "<<tm.MilliSeconds();
//      }
      for(int k = 0; k < sel.size(); k++) {
        if(sel[k]) {
          float bw = proposal_cur[k].x2 - proposal_cur[k].x1 + bsz01;
          float bh = proposal_cur[k].y2 - proposal_cur[k].y1 + bsz01;
          if(bw <= 0 || bh <= 0) continue;
          float bwxh = bw * bh;
//          for(int t = 0; t < top.size(); t++) {
          for(int t = 0; t < 1; t++) {
//            if(bwxh > this->proposal_min_area_vec_[t] && bwxh < this->proposal_max_area_vec_[t]) {
            if(bwxh > 6.160560 * 6.160560) {
              proposal_batch_vec[t].push_back(proposal_cur[k]);
            }
          }
        }
      }
//    }
  }

  float* top_boxes_scores;
//  for(int t = 0; t < top.size(); t++) {
  for(int t = 0; t < 1; t++) {
//    if(proposal_batch_vec[t].empty()) {
      // for special case when there is no box
//      top[t]->Reshape(1, rois_dim_, 1, 1);
//      float* top_boxes_scores = top[t]->mutable_cpu_data();
//      caffe_set(top[t]->count(), float(-1), top_boxes_scores);
//    } else {
      const int top_num = proposal_batch_vec[t].size();
//      top[t]->Reshape(top_num, rois_dim_, 1, 1);
//      float* top_boxes_scores = top[t]->mutable_cpu_data();
      top_boxes_scores = new float[top_n_ * 5];
      for (int j = 0; j < top_n_ * 5; ++j) {
        top_boxes_scores[j] = -1;
      }
      for (int k = 0; k < top_num; k++) {
        top_boxes_scores[k*rois_dim_] = proposal_batch_vec[t][k].id;
        top_boxes_scores[k*rois_dim_+1] = proposal_batch_vec[t][k].x1;
        top_boxes_scores[k*rois_dim_+2] = proposal_batch_vec[t][k].y1;
        top_boxes_scores[k*rois_dim_+3] = proposal_batch_vec[t][k].x2;
        top_boxes_scores[k*rois_dim_+4] = proposal_batch_vec[t][k].y2;
//        if (this->rpn_proposal_output_score_) {
//          top_boxes_scores[k*rois_dim_+5] = proposal_batch_vec[t][k].score;
//        }
      }
//    }
  }
#if DEBUG
  std::ofstream f_top;
  std::string f_top_str = "/apollo/debug_cpu/top.txt";
  f_top.open(f_top_str.c_str());
  for (int i = 0; i < top_n_ * 5; ++i) {
    if (i != 0 && i % 5 == 0) {
      f_top << std::endl;
    }
    f_top << top_boxes_scores[i] << " ";
  }
  f_top.close();
#endif

  BASE_CUDA_CHECK(cudaMemcpyAsync(out_rois, top_boxes_scores, out_rois_size * sizeof(float), cudaMemcpyHostToDevice, stream));

//
//  if (top.size() == 0) {
//    for (int class_id = 0; class_id < this->num_class_; ++class_id) {
//      vector<BBox<float> > & cur_box_list = this->all_candidate_bboxes_[class_id];
//      vector<BBox<float> > & cur_outbox_list = this->output_bboxes_[class_id];
//      if (this->nms_use_soft_nms_[class_id]) {
//        this->is_candidate_bbox_selected_ = caffe::soft_nms(cur_box_list,
//                                                            this->nms_overlap_ratio_[class_id], this->nms_top_n_[class_id],
//                                                            this->nms_max_candidate_n_[class_id], bbox_size_add_one_,
//                                                            this->nms_voting_[class_id], this->nms_vote_iou_[class_id]);
//      } else {
//        this->is_candidate_bbox_selected_ = caffe::nms(cur_box_list,
//                                                       this->nms_overlap_ratio_[class_id], this->nms_top_n_[class_id],
//                                                       false, this->nms_max_candidate_n_[class_id], bbox_size_add_one_,
//                                                       this->nms_voting_[class_id], this->nms_vote_iou_[class_id]);
//      }
//      cur_outbox_list.clear();
//      for (int i = 0; i < this->is_candidate_bbox_selected_.size(); ++i) {
//        if (this->is_candidate_bbox_selected_[i]) {
//          int id = im_width_scale.size() > 1 ? cur_box_list[i].id : 0;
//          CHECK_LT(id, im_width_scale.size());
//          cur_box_list[i].x1 = cur_box_list[i].x1 / im_width_scale[id] + cords_offset_x[id];
//          cur_box_list[i].y1 = cur_box_list[i].y1 / im_height_scale[id] + cords_offset_y[id];
//          cur_box_list[i].x2 = cur_box_list[i].x2 / im_width_scale[id] + cords_offset_x[id];
//          cur_box_list[i].y2 = cur_box_list[i].y2 / im_height_scale[id] + cords_offset_y[id];
//          cur_outbox_list.push_back(cur_box_list[i]);
//        }
//      }
//      cur_box_list.clear();
//    }
//  }

  return 0;
}
}  // namespace inference
}  // namespace perception
}  // namespace apollo