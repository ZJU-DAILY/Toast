import os
import tqdm
import pickle as pkl
from datetime import datetime

from utils.data import SPoint, STPoint, project_pt_to_road
from utils.data import Trajectory


class ParseTraj:
    """
    ParseTraj is an abstract class for parsing trajectory.
    It defines parse() function for parsing trajectory.
    """

    def __init__(self):
        pass

    def parse(self, input_path, is_target):
        """
        The parse() function is to load data to a list of Trajectory()
        """
        raise NotImplementedError


def create_datetime(timestamp):
    timestamp = str(timestamp)
    return datetime.fromtimestamp(int(timestamp))


class ParseRawTraj(ParseTraj):
    """
    Parse original GPS points to trajectories list. No extra data preprocessing
    """

    def __init__(self):
        super().__init__()

    def parse(self, input_path, is_target=None):
        """
        Args:
        -----
        input_path:
            str. input directory with file name
        Returns:
        --------
        trajs:
            list. list of trajectories. trajs contain input_path file's all gps points
        """
        with open(input_path, 'r') as f:
            trajs = []
            pt_list = []
            for line in tqdm.tqdm(f.readlines()):
                attrs = line.rstrip().split(' ')
                if attrs[0][0] == '-':
                    if len(pt_list) > 1:
                        traj = Trajectory(pt_list)
                        trajs.append(traj)
                    pt_list = []
                else:
                    lat = float(attrs[1])
                    lng = float(attrs[2])
                    pt = STPoint(lat, lng, create_datetime(str(attrs[0])))
                    # pt contains all the attributes of class STPoint
                    pt_list.append(pt)
            if len(pt_list) > 1:
                traj = Trajectory(pt_list)
                trajs.append(traj)
        return trajs


class ParseMMTraj(ParseTraj):
    """
    Parse map matched GPS points to trajectories list. No extra data preprocessing
    """
    def __init__(self, road_net):
        super().__init__()
        self.road_net = road_net

    def parse(self, input_path, is_target=True, is_save=False):
        """
        Args:
        -----
        input_path:
            str. input directory with file name
        Returns:
        --------
        trajs:
            list. list of trajectories. trajs contain input_path file's all gps points
        """
        pickle_path = input_path.replace(".txt", ".pkl")

        if os.path.exists(pickle_path):
            trajs = pkl.load(open(pickle_path, "rb"))
            return trajs

        with open(input_path, 'r') as f:
            trajs = []
            pt_list = []
            for line in tqdm.tqdm(f.readlines()):
                attrs = line.rstrip().split(' ')
                if attrs[0][0] == '-':
                    if len(pt_list) > 1:
                        traj = Trajectory(pt_list)
                        trajs.append(traj)
                    pt_list = []
                else:
                    lat = float(attrs[1])
                    lng = float(attrs[2])
                    rid = int(attrs[3])
                    if is_target:
                        projection, rate, dist = project_pt_to_road(self.road_net, SPoint(lat, lng), rid)
                        pt = STPoint(projection.lat, projection.lng, create_datetime(str(attrs[0])),
                                     segment_id=rid, rate=rate)
                        # candi_pt = CandidatePoint(projection.lat, projection.lng, rid, dist,
                        #                           rate * self.road_net.edgeDis[rid], rate)
                        # pt = STPoint(lat, lng, create_datetime(str(attrs[0])), {'candi_pt': candi_pt})
                    else:
                        pt = STPoint(lat, lng, create_datetime(str(attrs[0])))
                    # pt contains all the attributes of class STPoint
                    pt_list.append(pt)
            if len(pt_list) > 1:
                traj = Trajectory(pt_list)
                trajs.append(traj)

        if is_save:
            pickle_root = pickle_path
            while pickle_root[-1] != '/':
                pickle_root = pickle_root[:-1]
            if not os.path.exists(pickle_root):
                os.makedirs(pickle_root)
            pkl.dump(trajs, open(pickle_path, "wb+"))
        return trajs
