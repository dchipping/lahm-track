import sys
import random

import torch
from ray.rllib.agents import dqn, impala, ppo

JDE = __import__('Towards-Realtime-MOT')
sys.path.insert(0, JDE.__path__[0])

from models import *
from tracker import matching
from tracker.basetrack import BaseTrack, TrackState
from utils.kalman_filter import KalmanFilter
from utils.log import logger

from .jde_train import TrainAgentJdeTracker, AgentSTrack


class RandomAgent:
    @staticmethod
    def compute_single_action(obs):
        return random.randint(0, 1)


class GreedyAgent:
    @staticmethod
    def compute_single_action(obs):
        return 1


class AgentJdeTracker(TrainAgentJdeTracker):
    def __init__(self, opt, frame_rate=30, agent_path=None):
        self.opt = opt
        self.model = Darknet(opt.cfg, nID=14455)
        # load_darknet_weights(self.model, opt.weights)
        self.model.load_state_dict(torch.load(opt.weights, map_location='cpu')[
                                   'model'], strict=False)
        self.model.cuda().eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilter()

        self.agent = self.build_agent(agent_path)
        BaseTrack._count = 0  # THIS IS A HACK, CAN WE EXPORT PYTORCH WEIGHTS
        # AND USE INDEPENDENTLY OF RESETTING ENV AND CAUSING COUNT TO GO UP?

    def build_agent(self, agent_path):
        if not agent_path:
            return None
        elif agent_path == 'greedy':
            return GreedyAgent()
        elif agent_path == 'random':
            return RandomAgent()
        elif 'dqn' in agent_path:
            config = dqn.DEFAULT_CONFIG.copy()
            config["framework"] = "torch"
            trainer = dqn.DQNTrainer(
                config=config, env="motgym:BaseJdeEnv")
        elif 'ppo' in agent_path:
            config = ppo.DEFAULT_CONFIG.copy()
            config["framework"] = "torch"
            trainer = ppo.PPOTrainer(
                config=config, env="motgym:BaseJdeEnv")
        elif 'impala' in agent_path:
            config = impala.DEFAULT_CONFIG.copy()
            config["framework"] = "torch"
            trainer = impala.ImpalaTrainer(
                config=config, env="motgym:BaseJdeEnv")
        trainer.restore(agent_path)
        return trainer

    def update(self, im_blob, img0):
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

        self.frame_id += 1
        activated_starcks = []      # for storing active tracks, for the current frame
        # Lost Tracks whose detections are obtained in the current frame
        refind_stracks = []
        # The tracks which are not obtained in the current frame but are not removed.(Lost for some time lesser than the threshold for removing)
        lost_stracks = []
        removed_stracks = []

        # t1 = time.time()
        # ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            pred = self.model(im_blob)
        # pred is tensor of all the proposals (default number of proposals: 54264). Proposals have information associated with the bounding box and embeddings
        pred = pred[pred[:, :, 4] > self.opt.conf_thres]
        if len(pred) > 0:
            dets = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres)[0].cpu()
            # Final proposals are obtained in dets. Information of bounding box and embeddings also included
            # Next step changes the detection scales
            scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
        
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

            detections = [AgentSTrack(AgentSTrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f.numpy(), min_iou_score, agent=self.agent)
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
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # We have obtained a detection from a track which is not active, hence put the track in refind_stracks list
                track.re_activate(det, self.frame_id, new_id=False)
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
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
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
            unconfirmed[itracked].update(detections[idet], self.frame_id)
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
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        # If the tracks are lost for more frames than the threshold number, the tracks are removed.
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
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

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        # print('Final {} s'.format(t5-t4))
        return output_stracks


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
