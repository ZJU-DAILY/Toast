from typing import List
from datetime import timedelta

from utils.data.point import (
    SPoint,
    STPoint,
    haversine_distance,
    cal_loc_along_line
)


class Trajectory:
    """
    Trajectory creates a data type for trajectory, i.e. Trajectory().
    Trajectory is a list of STPoint.
    """

    def __init__(self, pt_list: List[STPoint]):
        """
        Args:
        -----
        pt_list:
            list of STPoint(lat, lng, time), containing the attributes of class STPoint
        """
        self.pt_list = pt_list

    def get_duration(self):
        """
        Get duration of a trajectory (pt_list) in seconds.
        """
        return (self.pt_list[-1].time - self.pt_list[0].time).total_seconds()

    def get_distance(self):
        """
        Get geographical distance of a trajectory (pt_list) in meters.
        """
        dist = 0.0
        pre_pt = self.pt_list[0]
        for pt in self.pt_list[1:]:
            tmp_dist = haversine_distance(pre_pt, pt)
            dist += tmp_dist
            pre_pt = pt
        return dist

    def get_avg_time_interval(self):
        """
        Calculate average time interval between two GPS points in one trajectory (pt_list) in seconds.
        """
        point_time_interval = []
        # How clever method! zip to get time interval

        for pre, cur in zip(self.pt_list[:-1], self.pt_list[1:]):
            point_time_interval.append((cur.time - pre.time).total_seconds())
        return sum(point_time_interval) / len(point_time_interval)

    def get_avg_distance_interval(self):
        """
        Calculate average distance interval between two GPS points in one trajectory (pt_list) in meters.
        """
        point_dist_interval = []
        for pre, cur in zip(self.pt_list[:-1], self.pt_list[1:]):
            point_dist_interval.append(haversine_distance(pre, cur))
        return sum(point_dist_interval) / len(point_dist_interval)

    def get_start_time(self):
        """
        Return the start time of the trajectory.
        """
        return self.pt_list[0].time

    def get_end_time(self):
        """
        Return the end time of the trajectory.
        """
        return self.pt_list[-1].time

    def get_mid_time(self):
        """
        Return the mid time of the trajectory.
        """
        return self.pt_list[0].time + (self.pt_list[-1].time - self.pt_list[0].time) / 2.0

    def get_centroid(self):
        """
        Get centroid SPoint.
        """
        mean_lat = 0.0
        mean_lng = 0.0
        for pt in self.pt_list:
            mean_lat += pt.lat
            mean_lng += pt.lng
        mean_lat /= len(self.pt_list)
        mean_lng /= len(self.pt_list)
        return SPoint(mean_lat, mean_lng)

    def query_trajectory_by_temporal_range(self, start_time, end_time):
        """
        Return the subtrajectory within start time and end time
        """
        # start_time <= pt.time < end_time
        traj_start_time = self.get_start_time()
        traj_end_time = self.get_end_time()
        if start_time > traj_end_time:
            return None
        if end_time <= traj_start_time:
            return None
        st = max(traj_start_time, start_time)
        et = min(traj_end_time + timedelta(seconds=1), end_time)
        start_idx = self.binary_search_idx(st)  # pt_list[start_idx].time <= st < pt_list[start_idx+1].time
        if self.pt_list[start_idx].time < st:
            # then the start_idx is out of the range, we need to increase it
            start_idx += 1
        end_idx = self.binary_search_idx(et)  # pt_list[end_idx].time <= et < pt_list[end_idx+1].time
        if self.pt_list[end_idx].time < et:
            # then the end_idx is acceptable
            end_idx += 1
        sub_pt_list = self.pt_list[start_idx:end_idx]
        return Trajectory(sub_pt_list)

    def binary_search_idx(self, time):
        # self.pt_list[idx].time <= time < self.pt_list[idx+1].time
        # if time < self.pt_list[0].time, return -1
        # if time >= self.pt_list[len(self.pt_list)-1].time, return len(self.pt_list)-1
        nb_pts = len(self.pt_list)
        if time < self.pt_list[0].time:
            return -1
        if time >= self.pt_list[-1].time:
            return nb_pts - 1
        # the time is in the middle
        left_idx = 0
        right_idx = nb_pts - 1
        while left_idx <= right_idx:
            mid_idx = int((left_idx + right_idx) / 2)
            if mid_idx < nb_pts - 1 and self.pt_list[mid_idx].time <= time < self.pt_list[mid_idx + 1].time:
                return mid_idx
            elif self.pt_list[mid_idx].time < time:
                left_idx = mid_idx + 1
            else:
                right_idx = mid_idx - 1

    def query_location_by_timestamp(self, time):
        """
        Return the GPS location given the time and trajectory (using linear interpolation).
        """
        idx = self.binary_search_idx(time)
        if idx == -1 or idx == len(self.pt_list) - 1:
            return None
        if self.pt_list[idx].time == time or (self.pt_list[idx + 1].time - self.pt_list[idx].time).total_seconds() == 0:
            return SPoint(self.pt_list[idx].lat, self.pt_list[idx].lng)
        else:
            # interpolate location
            dist_ab = haversine_distance(self.pt_list[idx], self.pt_list[idx + 1])
            if dist_ab == 0:
                return SPoint(self.pt_list[idx].lat, self.pt_list[idx].lng)
            dist_traveled = dist_ab * (time - self.pt_list[idx].time).total_seconds() / \
                            (self.pt_list[idx + 1].time - self.pt_list[idx].time).total_seconds()
            return cal_loc_along_line(self.pt_list[idx], self.pt_list[idx + 1], dist_traveled / dist_ab)

    def to_wkt(self):
        wkt = 'LINESTRING ('
        for pt in self.pt_list:
            wkt += '{} {}, '.format(pt.lng, pt.lat)
        wkt = wkt[:-2] + ')'
        return wkt

    def __hash__(self):
        return hash(self.pt_list[0].time.strftime('%Y%m%d%H%M%S') + '_' +
                    self.pt_list[-1].time.strftime('%Y%m%d%H%M%S'))

    def __eq__(self, other):
        return hash(self) == hash(other)
