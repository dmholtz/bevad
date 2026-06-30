import carla
import numpy as np

from bevad_sim.data_interface.data_types import BoundaryType, LaneType
from bevad_sim.data_interface.episode_map import EpisodeMap


def lateral_shift(transform, shift):
    """Makes a lateral shift of the forward vector of a transform"""
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()


### TODO: Refactor to new map


class CarlaMapConverter:
    @staticmethod
    def get_lane_marking(carla_marking):
        ltype = BoundaryType.UNKNOWN

        if carla_marking.color == carla.LaneMarkingColor.Yellow:
            if carla_marking.type == carla.LaneMarkingType.Solid:
                ltype = BoundaryType.SOLID_YELLOW

            if carla_marking.type == carla.LaneMarkingType.SolidSolid:
                ltype = BoundaryType.DOUBLE_SOLID_YELLOW

            if carla_marking.type == carla.LaneMarkingType.Broken:
                ltype = BoundaryType.DASHED_YELLOW

            if carla_marking.type == carla.LaneMarkingType.BrokenBroken:
                ltype = BoundaryType.DOUBLE_DASH_YELLOW

            if carla_marking.type == carla.LaneMarkingType.BrokenSolid:
                ltype = BoundaryType.DASH_SOLID_YELLOW

            if carla_marking.type == carla.LaneMarkingType.SolidBroken:
                ltype = BoundaryType.SOLID_DASH_YELLOW

            if carla_marking.type == carla.LaneMarkingType.BottsDots:
                ltype = BoundaryType.DASHED_YELLOW

        if carla_marking.color in (carla.LaneMarkingColor.White, carla.LaneMarkingColor.Standard):
            if carla_marking.type == carla.LaneMarkingType.Solid:
                ltype = BoundaryType.SOLID_WHITE

            if carla_marking.type == carla.LaneMarkingType.SolidSolid:
                ltype = BoundaryType.DOUBLE_SOLID_WHITE

            if carla_marking.type == carla.LaneMarkingType.Broken:
                ltype = BoundaryType.DASHED_WHITE

            if carla_marking.type == carla.LaneMarkingType.BrokenBroken:
                ltype = BoundaryType.DOUBLE_DASH_WHITE

            if carla_marking.type == carla.LaneMarkingType.BrokenSolid:
                ltype = BoundaryType.DASH_SOLID_WHITE

            if carla_marking.type == carla.LaneMarkingType.SolidBroken:
                ltype = BoundaryType.SOLID_DASH_WHITE

            if carla_marking.type == carla.LaneMarkingType.BottsDots:
                ltype = BoundaryType.DASHED_WHITE

        if ltype == BoundaryType.UNKNOWN and carla_marking.type == carla.LaneMarkingType.BottsDots:
            ltype = BoundaryType.DASHED_WHITE

        if carla_marking.type == carla.LaneMarkingType.Curb:
            ltype = BoundaryType.PHYSICAL

        if carla_marking.type == carla.LaneMarkingType.Grass:
            ltype = BoundaryType.GRASS

        if carla_marking.type == carla.LaneMarkingType.NONE:
            ltype = BoundaryType.NONE

        return ltype

    @staticmethod
    def convert_map(carla_map, lane_step_size=1.0):
        res = EpisodeMap.create_empty()
        CarlaMapConverter.extract_crosswalks(carla_map, res)
        ### TODO: Reimplement extract_traffic_control_elements!!!
        # CarlaMapConverter.extract_traffic_control_elements(carla_map, res)

        CarlaMapConverter.sample_center_lines(carla_map, res, lane_step_size)

        ### TODO: Hack: Flip y coordinates to convert from CARLA left-handed to right-handed coordinate systems.
        ### Is there a better solution?
        res.lane_centerlines[:, 1] *= -1
        res.lane_left_boundaries[:, 1] *= -1
        res.lane_right_boundaries[:, 1] *= -1
        res.road_segments[:, 1] *= -1
        res.driveable_space[:, 1] *= -1
        res.crosswalks[:, 1] *= -1
        res.parking_area[:, 1] *= -1

        return res

    @staticmethod
    def extract_crosswalks(carla_map, emap: EpisodeMap):
        crosswalks_points = carla_map.get_crosswalks()
        poly = []
        current_cw_id = 0
        for p in crosswalks_points:
            if p in poly:
                cwpoly = np.array([(tp.x, tp.y, tp.z, 1.0) for tp in poly])
                emap.add_crosswalk(cwpoly, current_cw_id)
                current_cw_id += 1
                poly = []
            else:
                poly.append(p)

    ### TODO: do we need landmarks or signs from landmarks?
    @staticmethod
    def extract_traffic_control_elements(carla_map, _emap: EpisodeMap):
        landmarks = carla_map.get_all_landmarks()
        print(landmarks)

    @staticmethod
    def sample_center_lines(carla_map, emap: EpisodeMap, lane_step_size: float):
        topology = carla_map.get_topology()
        # known_segments = set()
        lane_cids = {}
        cur_lane_id = 0
        for wp_tuple in topology:
            for wp in wp_tuple:
                lane_cid = (wp.road_id, wp.lane_id, wp.section_id)
                if lane_cid in lane_cids:
                    # print("segment: " + str(lane_cid) + " already known!")
                    continue
                # known_segments.add(lane_cid)

                lane_type = LaneType.UNKNOWN

                if wp.lane_type == carla.LaneType.Driving:
                    lane_type = LaneType.NORMAL

                points = wp.next_until_lane_end(lane_step_size)

                t_seg = []
                t_bound1 = []
                t_bound2 = []

                p = wp
                t_seg.append([p.transform.location.x, p.transform.location.y])
                b1 = lateral_shift(p.transform, 1.0 * p.lane_width * 0.5)
                b2 = lateral_shift(p.transform, -1.0 * p.lane_width * 0.5)
                t_bound1.append([b1.x, b1.y])
                t_bound2.append([b2.x, b2.y])

                for p in points:
                    t_seg.append([p.transform.location.x, p.transform.location.y])
                    b1 = lateral_shift(p.transform, 1.0 * p.lane_width * 0.5)
                    b2 = lateral_shift(p.transform, -1.0 * p.lane_width * 0.5)
                    t_bound1.append([b1.x, b1.y])
                    t_bound2.append([b2.x, b2.y])

                if len(t_seg) <= 1:
                    print("Lane segment too short: " + str(lane_cid))
                    break

                # segment = emap.LaneSegment(lane_type, np.array(t_seg), np.array(t_bound2), np.array(t_bound1))

                lbt = CarlaMapConverter.get_lane_marking(wp.left_lane_marking)
                rbt = CarlaMapConverter.get_lane_marking(wp.right_lane_marking)

                emap.add_segment(
                    np.array(t_seg), np.array(t_bound2), np.array(t_bound1), lane_type, lbt, rbt, cur_lane_id, 50.0
                )
                lane_cids[lane_cid] = cur_lane_id
                cur_lane_id += 1

        for wp_tuple in topology:
            src = wp_tuple[0]
            dst = wp_tuple[1]

            src_id = (src.road_id, src.lane_id, src.section_id)
            dst_id = (dst.road_id, dst.lane_id, dst.section_id)

            emap.set_lane_continue(lane_cids[src_id], lane_cids[dst_id])
