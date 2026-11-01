import numpy as np
from .vod_dataset_temporal import vodDataset_Temporal


class VoDDatasetVelDecomp(vodDataset_Temporal):
    """
    Extends temporal VoD loader with Doppler velocity decomposition.
    Appends vr_x = vrc * cos(atan2(y,x))
            vr_y = vrc * sin(atan2(y,x))
    as two new channels after the original 7.
    """

    @staticmethod
    def _append_vr_decomp(points):
        # Keep temporal index in the last column for temporal collation logic.
        # Input:  x,y,z,rcs,v_r,v_r_comp,time,tem_idx
        # Output: x,y,z,rcs,v_r,v_r_comp,time,vr_x,vr_y,tem_idx
        x = points[:, 0]
        y = points[:, 1]
        vrc = points[:, 5]  # v_r_comp
        azimuth = np.arctan2(y, x)
        vr_x = (vrc * np.cos(azimuth))[:, np.newaxis]
        vr_y = (vrc * np.sin(azimuth))[:, np.newaxis]
        base = points[:, :7]
        tem_idx = points[:, 7:8]
        return np.concatenate([base, vr_x, vr_y, tem_idx], axis=1)

    def get_temporal_radar(self, idx, info):
        radar_list, temporal_list = super().get_temporal_radar(idx, info)
        radar_list = [self._append_vr_decomp(points) for points in radar_list]
        return radar_list, temporal_list
