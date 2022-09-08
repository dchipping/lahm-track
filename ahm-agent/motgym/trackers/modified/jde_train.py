import sys
from collections import deque

from scipy.spatial.distance import cdist

JDE = __import__('Towards-Realtime-MOT')
sys.path.insert(0, JDE.__path__[0])

from models import *
from numba import jit
from tracker import matching
from utils.kalman_filter import KalmanFilter
from utils.log import logger

from tracker.basetrack import BaseTrack, TrackState


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
        self.curr_feat = np.asarray(temp_feat, dtype=float)

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

        # Recalculate gallery each update same as baseline FairMOT
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * \
                self.smooth_feat + (1-self.alpha) * feat
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

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
    def multi_predict(stracks, kalman_filter):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
#            multi_mean, multi_covariance = STrack.kalman_filter.multi_predict(multi_mean, multi_covariance)
            multi_mean, multi_covariance = kalman_filter.multi_predict(multi_mean, multi_covariance)
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


class TrainAgentJdeTracker():
    def __init__(self, opt, frame_rate=30):
        self.opt = opt

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilter()

    def update(self, dets, frame_id):
        """
        Processes the image frame and finds bounding box(detections).

        Associates the detection with corresponding tracklets and also handles lost, removed, refound and active tracklets

        Parameters
        ----------
        im_blob : torch.float32
                  Tensor of shape depending upon the size of image. By default, shape of this tensor is [1, 3, 608, 1088]

        img0 : ndarray
               ndarray of shape depending on the input image sequence. By default, shape is [608, 1080, 3]

        Returns
        -------
        output_stracks : list of Strack(instances)
                         The list contains information regarding the online_tracklets for the recieved image tensor.

        """
        activated_starcks = []      # for storing active tracks, for the current frame
        # Lost Tracks whose detections are obtained in the current frame
        refind_stracks = []
        # The tracks which are not obtained in the current frame but are not removed.(Lost for some time lesser than the threshold for removing)
        lost_stracks = []
        removed_stracks = []

        # t1 = time.time()
        # ''' Step 1: Network forward, get detections & embeddings'''
        # with torch.no_grad():
        #     pred = self.model(im_blob)
        # # pred is tensor of all the proposals (default number of proposals: 54264). Proposals have information associated with the bounding box and embeddings
        # pred = pred[pred[:, :, 4] > self.opt.conf_thres]
        # if len(pred) > 0:
        #     dets = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres)[0].cpu()
        #     # Final proposals are obtained in dets. Information of bounding box and embeddings also included
        #     # Next step changes the detection scales
        #     scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
        
        # pred now has lesser number of proposals. Proposals rejected on basis of object confidence score
        if len(dets) > 0:
            '''Detections is list of (x1, y1, x2, y2, object_conf, class_score, class_pred)'''
            # class_pred is the embeddings.

            # Calculate maximum overlap of each detection with neighbouring detections
            # Lower score means more overlap, min iou score = worst overlap
            if len(dets) > 1:
                min_iou_scores = get_min_iou_scores(dets[:, :4])
            else:
                min_iou_scores = [1.]

            detections = [AgentSTrack(AgentSTrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], np.asarray(f), min_iou_score, agent=None)
                          for (tlbrs, f, min_iou_score) in zip(dets[:, :5], dets[:, 6:], min_iou_scores)]
        else:
            detections = []

        t2 = time.time()
        # print('Forward: {} s'.format(t2-t1))

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                # previous tracks which are not active in the current frame are added in unconfirmed list
                unconfirmed.append(track)
                # print("Should not be here, in unconfirmed")
            else:
                # Active tracks are added to the local list 'tracked_stracks'
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        # Combining currently tracked_stracks and lost_stracks
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        AgentSTrack.multi_predict(strack_pool, self.kalman_filter)

        dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        dists = matching.fuse_motion(
            self.kalman_filter, dists, strack_pool, detections)
        # The dists is the list of distances of the detection with the tracks in strack_pool
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=0.7)
        # The matches is the array for corresponding matches of the detection with the corresponding strack_pool

        for itracked, idet in matches:
            # itracked is the id of the track and idet is the detection
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # If the track is active, add the detection to the track
                track.update(detections[idet], frame_id)
                activated_starcks.append(track)
            else:
                # We have obtained a detection from a track which is not active, hence put the track in refind_stracks list
                track.re_activate(det, frame_id, new_id=False)
                refind_stracks.append(track)

        # None of the steps below happen if there are no undetected tracks.
        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        # detections is now a list of the unmatched detections
        r_tracked_stracks = []  # This is container for stracks which were tracked till the
        # previous frame but no detection was found for it in the current frame
        for i in u_track:
            if strack_pool[i].state == TrackState.Tracked:
                r_tracked_stracks.append(strack_pool[i])
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=0.5)
        # matches is the list of detections which matched with corresponding tracks by IOU distance method
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, frame_id, new_id=False)
                refind_stracks.append(track)
        # Same process done for some unmatched detections, but now considering IOU_distance as measure

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        # If no detections are obtained for tracks (u_track), the tracks are added to lost_tracks list and are marked lost

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], frame_id)
            activated_starcks.append(unconfirmed[itracked])

        # The tracks which are yet not matched
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # after all these confirmation steps, if a new detection is found, it is initialized for a new track
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        # If the tracks are lost for more frames than the threshold number, the tracks are removed.
        for track in self.lost_stracks:
            if frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        # print('Remained match {} s'.format(t4-t3))

        # Update the self.tracked_stracks and self.lost_stracks using the updates in this step.
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, refind_stracks)
        # self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.lost_stracks = sub_stracks(
            self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(
            self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)

        # get scores of lost tracks
        output_stracks = [
            track for track in self.tracked_stracks if track.is_activated]

        # logger.debug('===========Frame {}=========='.format(self.frame_id))
        # logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        # logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        # logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        # logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        # print('Final {} s'.format(t5-t4))
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
