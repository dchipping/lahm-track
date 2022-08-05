from collections import deque

import numpy as np
from scipy.spatial.distance import cdist

import FairMOT.src._init_paths
from tracker import matching
from tracker.basetrack import BaseTrack, TrackState
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.utils import *


class AgentSTrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, min_iou_score, agent=None):
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.tracklet_len = 0

        self.agent = agent  # If this is None gallery will be frozen

        self.score = score
        self.min_iou_score = min_iou_score
        self.curr_feat = temp_feat

        self.smooth_feat = temp_feat
        self.features = deque([])
        self.alpha = 0.9

    def gallery_similarity(self, feat):
        max_cosine_simlarity = 0.
        if self.features:
            dists = cdist(np.array([feat]), np.asarray(
                self.features), 'cosine')
            max_cosine_simlarity = np.max(np.ones(dists.shape) - dists)
        return max_cosine_simlarity

    def average_gallery_distance(self):
        average_dist = 0.
        n = len(self.features)
        if n > 2:
            feats = np.asarray(self.features)
            cost_matrix = cdist(feats, feats, 'cosine')
            pairwise_dists = np.tril(cost_matrix, -1)
            num_pairs = (n-1)*(n)/2
            average_dist = np.sum(pairwise_dists) / num_pairs
        return average_dist

    def get_observation(self, feat, score, min_iou_score):
        similarity = self.gallery_similarity(feat)
        average = 0.
        if self.features:
            gallery_avg = np.average(self.features, axis=0)
            average = cdist([gallery_avg], [feat], 'cosine').item()
        average_dist = self.average_gallery_distance()
        return np.array([
            score,
            similarity,
            len(self.features),
            min_iou_score,
            average,
            average_dist
        ], dtype=float)

    def update_gallery(self, action, feat):
        '''Translate action to change in gallery'''
        if action == 0:
            return
        elif action == 1:
            self.features.append(feat)
        elif action == -1:
            self.prune_similar()

        # Recalculate gallery each update same as baseline FairMOT
        self.smooth_feat = None  # Reset smooth_feature
        for i, feat in enumerate(self.features):
            if self.smooth_feat is None:
                self.smooth_feat = feat
            else:
                feat /= np.linalg.norm(feat)
                self.smooth_feat = self.alpha * \
                    self.smooth_feat + (1 - self.alpha) * feat
            self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def prune_similar(self):
        # Remove most similar feature from gallery
        gallery = np.asarray(self.features)
        n, _ = gallery.shape
        cost_matrix = np.eye(n) + cdist(gallery, gallery, 'cosine')
        flatten_idx = np.argmin(cost_matrix)
        track1_idx, track2_idx = np.unravel_index(flatten_idx, (n, n))
        self.features.pop(track1_idx)

    def agent_update_features(self, feat, obs):
        '''New method added for RL agent to manage gallery'''
        self.curr_feat = feat
        if self.agent:
            action = self.agent.compute_single_action(obs)
            self.update_gallery(action, feat)

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.obs = self.get_observation(
            new_track.curr_feat,
            new_track.score,
            new_track.min_iou_score
        )
        self.agent_update_features(new_track.curr_feat, self.obs)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.obs = self.get_observation(
            new_track.curr_feat,
            new_track.score,
            new_track.min_iou_score
        )
        self.agent_update_features(new_track.curr_feat, self.obs)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = AgentSTrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        self.obs = self.get_observation(
            self.curr_feat,
            self.score,
            self.min_iou_score
        )
        self.agent_update_features(self.curr_feat, self.obs)

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class TrainAgentJDETracker():
    def __init__(self, opt, frame_rate=30, lookup_gallery=0):
        self.opt = opt
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()
        self.lookup_gallery = lookup_gallery

    def reset(self):
        BaseTrack._count = 0
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.kalman_filter = KalmanFilter()

    def update(self, dets, id_feature, frame_id):
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        ### Detections and features are pre-generated using gen_fairmot_jde.py ###

        # Calculate maximum overlap of each detection with neighbouring detections
        # Lower score means more overlap, min iou score = worst overlap
        min_iou_scores = get_min_iou_scores(dets)

        if len(dets) > 0:
            '''Detections'''
            detections = [AgentSTrack(AgentSTrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, min_iou_score, agent=None)
                          for (tlbrs, f, min_iou_score) in zip(dets[:, :5], id_feature, min_iou_scores)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        # for strack in strack_pool:
        # strack.predict()
        AgentSTrack.multi_predict(strack_pool)
        if self.lookup_gallery:
            dists = custom_embedding_distance(
                self.lookup_gallery, strack_pool, detections)
        else:
            dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_motion(
            self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=0.4)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i]
                             for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(
            self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(
            self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)

        # logger.debug('===========Frame {}=========='.format(frame_id))
        # logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        # logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        # logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        # logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return self.tracked_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


def custom_embedding_distance(n, tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray(
        [track.curr_feat for track in detections], dtype=np.float)

    # Pick n min from gallery as smooth_feat (in place of moving average)
    num_dets = len(detections)
    track_features = [] # (num_tracks, num_dets, 128)
    for i, track in enumerate(tracks):
        if track.features:
            feats = np.asarray(track.features)

            # If gallery is larger than n, take best n features
            if len(track.features) > n:
                gallery_cost_matrix = cdist(feats, det_features, metric)
                min_row_idxs = np.argpartition(
                    gallery_cost_matrix, n, axis=0)[:n].T
                assert min_row_idxs.shape == (num_dets, n)

                # Best averaged track feature for each detection
                smooth_feat_vs_det = np.empty(
                    (num_dets, 128))  # (num_dets, 128)
                for det_n in range(num_dets):
                    best_n_feats = feats[min_row_idxs[det_n], :]  # (n, 128)
                    smooth_feat = np.average(best_n_feats, axis=0)  # (128,)
                    smooth_feat_vs_det[det_n, :] = smooth_feat
            else:
                smooth_feat = np.average(feats, axis=0)
                smooth_feat_vs_det = np.tile(smooth_feat, (num_dets, 1))

            track_features.append(smooth_feat_vs_det)

    for det_n in range(num_dets):
        track_features_vs_det = []
        for track_n in range(len(tracks)):
            feat = track_features[track_n][det_n]
            track_features_vs_det.append(feat)

        cost_col = cdist(track_features_vs_det, [det_features[det_n]], metric)
        cost_matrix[:, det_n] = cost_col.flatten()

    cost_matrix = np.maximum(0.0, cost_matrix) # Nomalized features
    return cost_matrix


def get_min_iou_scores(dets):
    # Calculate maximum overlap of each detection with neighbouring detections
    # Lower score means more overlap, min iou score = worst overlap
    min_iou_scores = []
    n_dets = dets.shape[0]
    for idx in range(n_dets):
        mask = np.ones((n_dets,), dtype=bool)
        mask[idx] = False
        neighbours_dets = dets[mask, :]
        ious = matching.iou_distance(np.expand_dims(
            dets[idx, :], axis=0), neighbours_dets)
        min_iou_scores.append(np.min(ious))
    return min_iou_scores
