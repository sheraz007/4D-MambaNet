import copy
import torch
from .detector3d_template import Detector3DTemplate
from ..dense_heads.anchor_head_single import AnchorHeadSingle
from ..model_utils import model_nms_utils
from ..backbones_3d import vfe

class VoxelNeXtDoubleHead(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        #model_cfg.pop('ROI_HEAD')
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list: list = self.build_networks()
        self.use_motion_loss = model_cfg.get('use_motion_loss', False)
        self.just_anchorcar_loss = model_cfg.get('just_anchorcar_loss', False)
        if model_cfg.get('DENSE_HEAD2', False):
            if self.model_cfg.get('DENSE_HEAD', False):
                dense_head1_bak = copy.deepcopy(self.model_cfg.DENSE_HEAD)
            else:
                dense_head1_bak = self.model_cfg.DENSE_HEAD2
            self.model_cfg.DENSE_HEAD = self.model_cfg.DENSE_HEAD2
            module, _ = super().build_dense_head(self.model_info_dict)
            self.add_module('dense_head2', module)
            self.model_cfg.DENSE_HEAD = dense_head1_bak if 'DENSE_HEAD' in self.model_cfg else None
        if model_cfg.get('ROI_HEAD', False):
            self.module_list[-2], self.module_list[-1] = self.module_list[-1], self.module_list[-2]

        if model_cfg.get('VFE2', False):
            vfe_module = vfe.__all__[self.model_cfg.VFE2.NAME](
                model_cfg=self.model_cfg.VFE2,
                num_point_features=self.model_info_dict['num_rawpoint_features'],
                point_cloud_range=self.model_info_dict['point_cloud_range'],
                voxel_size=self.model_info_dict['voxel_size'],
                grid_size=self.model_info_dict['grid_size'],
                depth_downsample_factor=self.model_info_dict['depth_downsample_factor']
            )
            self.add_module('vfe2', vfe_module)
            self.module_list.insert(1, vfe_module)
    
        #self.model_cfg.POST_PROCESSING.FILTER_OUT_EMPTY_BOX = True
        self.post_processing_type = self.model_cfg.get('POST_PROCESSING_TYPE', 'anchorcar_vxrest')
        self.just_vx_head = model_cfg.get('just_vx_head', False)
        self.only_train_roi_head = self.model_cfg.get('ONLY_TRAIN_ROI_HEAD', False)
        if self.only_train_roi_head:
            for module in self.module_list[:-1]:
                module: torch.nn.Module
                for p in module.parameters():
                    p.requires_grad_(False)

    def forward(self, batch_dict):
        
        if self.only_train_roi_head:
            with torch.no_grad():
                for cur_module in self.module_list[:-1]:
                    cur_module.eval()
                    batch_dict = cur_module(batch_dict)
            batch_dict = self.module_list[-1](batch_dict)
        else:
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)

            # for debug 看看validation loss收不收敛
            if self.model_cfg.get('TRACK_VAL_LOSS', False):
                loss, tb_dict, disp_dict = self.get_training_loss()
                recall_dicts['val_loss'] = {}
                recall_dicts['val_loss'].update(tb_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        if not self.only_train_roi_head:
            loss, tb_dict = 0, dict()

            if self.use_motion_loss:
                motion_loss = self.feat_tem.get_motion_loss()
                loss += motion_loss
                tb_dict['motion_loss'] = motion_loss
            if self.dense_head is not None:
                loss_head1, tb_dict = self.dense_head.get_loss(tb_dict)    
                loss += loss_head1
            if self.just_vx_head == False:
                loss_head2, tb_dict2 = self.dense_head2.get_loss()
                loss += loss_head2
                tb_dict2 = dict((k+'_2', v) for k, v in tb_dict2.items())
                tb_dict.update(tb_dict2)
            if self.roi_head is not None:
                loss_roi, tb_dict = self.roi_head.get_loss(tb_dict)
                loss += loss_roi

            if self.just_anchorcar_loss:
                loss = loss_head2
                tb_dict = tb_dict2
        else:
            loss, tb_dict = self.roi_head.get_loss()
        
        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
        
        return loss, tb_dict, disp_dict
    
    def post_processing(self, batch_dict):
        if self.post_processing_type == 'anchor_head':
            return super().post_processing(batch_dict)
        elif self.post_processing_type == 'vx_head':
            return self.post_processing_vxhead(batch_dict)
        elif self.post_processing_type == 'anchorcar_vxrest':
            return self.post_processing_anchorcar_vxrest(batch_dict)
        elif self.post_processing_type == 'all_combine':
            return self.post_processing_all_combine(batch_dict)
        elif self.post_processing_type == 'double_head':
            return self.post_processing_double_head(batch_dict)
        elif self.post_processing_type == 'roi_head':
            return super().post_processing(batch_dict)
        else:
            raise NotImplementedError
        
    def post_processing_anchorcar_vxrest(self, batch_dict):
        pred_dict_anchor, recall_dict_anchor = super().post_processing(batch_dict)
        pred_dict_vx, recall_dict  = self.post_processing_vxhead(batch_dict)

        if self.just_anchorcar_loss:
            recall_dict = recall_dict_anchor
        batch_size = batch_dict['batch_size']
        pred_dict = []
        for i in range(batch_size):
            mask_ac = torch.logical_or(pred_dict_anchor[i]['pred_labels']==1, pred_dict_anchor[i]['pred_labels']==4)
            mask_vx = torch.logical_and(pred_dict_vx[i]['pred_labels']!=1, pred_dict_vx[i]['pred_labels']!=4)

            if self.just_anchorcar_loss:
                mask_vx = torch.tensor([False]).repeat(pred_dict_vx[i]['pred_boxes'].shape[0])
                
            pred_dict_batch = dict()
            pred_dict_batch['pred_boxes'] = torch.cat([pred_dict_vx[i]['pred_boxes'][mask_vx], pred_dict_anchor[i]['pred_boxes'][mask_ac]], dim=0)
            pred_dict_batch['pred_scores'] = torch.cat([pred_dict_vx[i]['pred_scores'][mask_vx], pred_dict_anchor[i]['pred_scores'][mask_ac]], dim=0)
            pred_dict_batch['pred_labels'] = torch.cat([pred_dict_vx[i]['pred_labels'][mask_vx], pred_dict_anchor[i]['pred_labels'][mask_ac]], dim=0)
            pred_dict.append(pred_dict_batch)
        return pred_dict, recall_dict
    
    def post_processing_double_head(self, batch_dict):
        batch_size = batch_dict['batch_size']
        pred_dicts = []
        recall_dict = {}
        for i in range(batch_size):
            pred_boxes_ac = batch_dict['pred_boxes_ac'][i]
            pred_scores_ac = batch_dict['pred_scores_ac'][i]
            pred_labels_ac = batch_dict['pred_labels_ac'][i]+1
            pred_boxes_vx = batch_dict['pred_boxes_vx'][i]
            pred_scores_vx = batch_dict['pred_scores_vx'][i]
            pred_labels_vx = batch_dict['pred_labels_vx'][i]+1
            car_mask_ac = pred_labels_ac!=-1#pred_labels_ac==1
            car_mask_vx = (pred_labels_vx!=1)# & (pred_labels_vx!=2)
            pred_boxes = torch.cat([pred_boxes_ac[car_mask_ac], pred_boxes_vx[car_mask_vx]])
            pred_scores = torch.cat([pred_scores_ac[car_mask_ac], pred_scores_vx[car_mask_vx]])
            pred_labels = torch.cat([pred_labels_ac[car_mask_ac], pred_labels_vx[car_mask_vx]])
            # pred_boxes = pred_boxes_vx[car_mask_vx]
            # pred_scores = pred_scores_vx[car_mask_vx]
            # pred_labels = pred_labels_vx[car_mask_vx]
            # pred_boxes = pred_boxes_vx
            # pred_scores = pred_scores_vx
            # pred_labels = pred_labels_vx
            # pred_boxes = pred_boxes_ac
            # pred_scores = pred_scores_ac
            # pred_labels = pred_labels_ac
            # (400, 7)
            # pred_boxes = torch.cat([pred_boxes_vx, pred_boxes_ac])
            # pred_scores = torch.cat([pred_scores_vx, pred_scores_ac])
            # pred_labels = torch.cat([pred_labels_vx, pred_labels_ac])
            multi_class_nms = False
            if multi_class_nms:
                final_scores = []
                final_labels = []
                final_boxes = []
                for k in range(1, 4):
                    class_mask = (pred_labels == k)
                    if class_mask.sum() == 0:
                        continue
                    pred_boxes_class = pred_boxes[class_mask]
                    pred_scores_class = pred_scores[class_mask]
                    pred_labels_class = pred_labels[class_mask]
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=pred_scores_class, box_preds=pred_boxes_class,
                        nms_config=self.model_cfg.POST_PROCESSING.NMS_CONFIG,
                        score_thresh=self.model_cfg.POST_PROCESSING.SCORE_THRESH
                    )
                    final_scores.append(selected_scores)
                    final_labels.append(pred_labels_class[selected])
                    final_boxes.append(pred_boxes_class[selected])
                final_scores = torch.cat(final_scores)
                final_boxes = torch.cat(final_boxes)
                final_labels = torch.cat(final_labels)
            else:
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=pred_scores, box_preds=pred_boxes,
                    nms_config=self.model_cfg.POST_PROCESSING.NMS_CONFIG,
                    score_thresh=self.model_cfg.POST_PROCESSING.SCORE_THRESH
                )
                final_scores = selected_scores
                final_labels = pred_labels[selected]
                final_boxes = pred_boxes[selected]
            recall_dict = self.generate_recall_record(
                box_preds=final_boxes,
                recall_dict=recall_dict, batch_index=i, data_dict=batch_dict,
                thresh_list=self.model_cfg.POST_PROCESSING.RECALL_THRESH_LIST
            )
            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)
        filter_out_empty_boxes = True
        points = batch_dict['points']
        if filter_out_empty_boxes:
            assert(points is not None)
            from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
            for k, final_dict in enumerate(pred_dicts):
                points_batch = points[points[:,0].long()==k]
                box_idxs_of_pts = points_in_boxes_gpu(points_batch[None,:,1:4], final_dict['pred_boxes'][None,:,:]).squeeze()
                selected = []
                for i in range(final_dict['pred_boxes'].shape[0]):
                    points_num_in_box = (box_idxs_of_pts == i).sum()
                    if points_num_in_box != 0:
                        selected.append(i)
                pred_dicts[k]['pred_boxes'] = pred_dicts[k]['pred_boxes'][selected]
                pred_dicts[k]['pred_scores'] = pred_dicts[k]['pred_scores'][selected]
                pred_dicts[k]['pred_labels'] = pred_dicts[k]['pred_labels'][selected]
        return pred_dicts, recall_dict

    def post_processing_all_combine(self, batch_dict):
        batch_size = batch_dict['batch_size']
        pred_dicts = []
        recall_dict = {}
        for i in range(batch_size):
            pred_boxes_ac = batch_dict['batch_box_preds'][i]
            pred_cls_ac = batch_dict['batch_cls_preds'][i].sigmoid()
            pred_scores_ac, pred_labels_ac = torch.max(pred_cls_ac, dim=-1)
            pred_labels_ac = pred_labels_ac + 1 
            scores_mask = (pred_scores_ac >= self.model_cfg.POST_PROCESSING.SCORE_THRESH)
            pred_boxes_ac = pred_boxes_ac[scores_mask]
            pred_scores_ac = pred_scores_ac[scores_mask]
            pred_labels_ac = pred_labels_ac[scores_mask]
            pred_boxes_vx = batch_dict['final_box_dicts'][i]['pre_nms_pred_boxes']
            pred_scores_vx = batch_dict['final_box_dicts'][i]['pre_nms_pred_scores']
            pred_labels_vx = batch_dict['final_box_dicts'][i]['pre_nms_pred_labels']
            # car_mask_ac = pred_labels_ac==1
            # car_mask_vx = pred_labels_vx!=1
            # pred_boxes = torch.cat([pred_boxes_ac[car_mask_ac], pred_boxes_vx[car_mask_vx]])
            # pred_scores = torch.cat([pred_scores_ac[car_mask_ac], pred_scores_vx[car_mask_vx]])
            # pred_labels = torch.cat([pred_labels_ac[car_mask_ac], pred_labels_vx[car_mask_vx]])
            pred_boxes = pred_boxes_vx
            pred_scores = pred_scores_vx
            pred_labels = pred_labels_vx
            selected, selected_scores = model_nms_utils.class_agnostic_nms(
                box_scores=pred_scores, box_preds=pred_boxes,
                nms_config=self.model_cfg.POST_PROCESSING.NMS_CONFIG,
                score_thresh=self.model_cfg.POST_PROCESSING.SCORE_THRESH
            )
            final_scores = selected_scores
            final_labels = pred_labels[selected]
            final_boxes = pred_boxes[selected]
            recall_dict = self.generate_recall_record(
                box_preds=final_boxes,
                recall_dict=recall_dict, batch_index=i, data_dict=batch_dict,
                thresh_list=self.model_cfg.POST_PROCESSING.RECALL_THRESH_LIST
            )
            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)
        filter_out_empty_boxes = True
        points = batch_dict['points']
        if filter_out_empty_boxes:
            assert(points is not None)
            from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
            for k, final_dict in enumerate(pred_dicts):
                points_batch = points[points[:,0].long()==k]
                box_idxs_of_pts = points_in_boxes_gpu(points_batch[None,:,1:4], final_dict['pred_boxes'][None,:,:]).squeeze()
                selected = []
                for i in range(final_dict['pred_boxes'].shape[0]):
                    points_num_in_box = (box_idxs_of_pts == i).sum()
                    if points_num_in_box != 0:
                        selected.append(i)
                pred_dicts[k]['pred_boxes'] = pred_dicts[k]['pred_boxes'][selected]
                pred_dicts[k]['pred_scores'] = pred_dicts[k]['pred_scores'][selected]
                pred_dicts[k]['pred_labels'] = pred_dicts[k]['pred_labels'][selected]
        return pred_dicts, recall_dict

    def post_processing_vxhead(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
