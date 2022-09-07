from ..sequential_env import SequentialFairmotEnv


class Mot17SequentialEnvSeq02(SequentialFairmotEnv):
    def __init__(self):
        dataset = 'MOT17/train_half'
        detections = 'FairMOT/MOT17/train_half'
        super().__init__(dataset, detections)
        self.seq = 'MOT17-02'
        self.assign_target()


class Mot17SequentialEnvSeq04(SequentialFairmotEnv):
    def __init__(self):
        dataset = 'MOT17/train_half'
        detections = 'FairMOT/MOT17/train_half'
        super().__init__(dataset, detections)
        self.seq = 'MOT17-04'
        self.assign_target()


class Mot17SequentialEnvSeq05(SequentialFairmotEnv):
    def __init__(self):
        dataset = 'MOT17/train_half'
        detections = 'FairMOT/MOT17/train_half'
        super().__init__(dataset, detections)
        self.seq = 'MOT17-05'
        self.assign_target()


class Mot17SequentialEnvSeq09(SequentialFairmotEnv):
    def __init__(self):
        dataset = 'MOT17/train_half'
        detections = 'FairMOT/MOT17/train_half'
        super().__init__(dataset, detections)
        self.seq = 'MOT17-09'
        self.assign_target()


class Mot20SequentialEnvSeq01(SequentialFairmotEnv):
    def __init__(self):
        dataset = 'MOT20/train_half'
        detections = 'FairMOT/MOT20/train_half'
        super().__init__(dataset, detections)
        self.seq = 'MOT20-01'
        self.assign_target()


class Mot20SequentialEnvSeq02(SequentialFairmotEnv):
    def __init__(self):
        dataset = 'MOT20/train_half'
        detections = 'FairMOT/MOT20/train_half'
        super().__init__(dataset, detections)
        self.seq = 'MOT20-02'
        self.assign_target()


class Mot20SequentialEnvSeq03(SequentialFairmotEnv):
    def __init__(self):
        dataset = 'MOT20/train_half'
        detections = 'FairMOT/MOT20/train_half'
        super().__init__(dataset, detections)
        self.seq = 'MOT20-03'
        self.assign_target()


class Mot20SequentialEnvSeq04(SequentialFairmotEnv):
    def __init__(self):
        dataset = 'MOT20/train_half'
        detections = 'FairMOT/MOT20/train_half'
        super().__init__(dataset, detections)
        self.seq = 'MOT20-04'
        self.assign_target()
