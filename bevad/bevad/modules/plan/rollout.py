import math

import torch
import torch.nn as nn


class BezierTrajectoryRollout(nn.Module):
    """Roll-out a trajectory given a speed-time along a polyline using a Bezier curve fit."""

    def __init__(
        self,
        n_bezier: int,
        polyline_len: int,
        input_frame_rate: int,
        output_frame_rate: int,
    ):
        super().__init__()

        self.n_bezier = n_bezier  # the degree of the Bezier curve

        self.temporal_upsampling = int(output_frame_rate / input_frame_rate)
        self.delta_t = 1 / output_frame_rate

        self._precompute_bezier_fit_lse(n_fit=polyline_len, n_bezier=n_bezier)

        # build sequence of binomial coefficients from "n choose 0" up to "n choose n"n
        binomial_coeffs = torch.tensor(
            [math.comb(self.n_bezier, i) for i in range(self.n_bezier + 1)],
            dtype=torch.float32,
        )
        self.register_buffer("binomial_coeffs", binomial_coeffs)

    def roll_out_speed(
        self,
        polyline: torch.Tensor,
        pred_speed: torch.Tensor,
        current_speed: torch.Tensor,
        planning_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            polyline (torch.Tensor): The predicted trajectory. Shape: B x N x 2
            speed (torch.Tensor): The sequence of future speed values. Shape: B x T
            current_speed (torch.Tensor): Shape: B x 1
            planning_mask (torch.Tensor | None): Optional mask, highlighting valid future timesteps. Shape: B x T
        """
        # compute the progress along the Bezier curve given the speed profile
        speed = torch.cat((current_speed, pred_speed), dim=-1)  # B x (T+1)
        mean_interval_speed = (speed[:, :-1] + speed[:, 1:]) / 2  # B x T
        mean_interval_speed = mean_interval_speed.repeat_interleave(
            self.temporal_upsampling, dim=1
        )  # B x T'
        progress = torch.cumsum(mean_interval_speed * self.delta_t, dim=1)  # B x T'

        return self.forward(polyline, progress, planning_mask)

    def roll_out_progress(self, polyline, progress, planning_mask=None):
        self.forward(polyline, progress, planning_mask)

    def forward(
        self,
        polyline: torch.Tensor,
        progress: torch.Tensor,
        planning_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # compute the Bezier time parameters t
        length = float(polyline.shape[1])  # TODO: calculate this value
        t = progress / length  # B x T'
        valid_mask = t <= 1  # B x T'

        # fit a Bezier curve to the polyline
        exp_polyline = torch.cat(
            (torch.zeros((polyline.shape[0], 1, 2), device=polyline.device), polyline),
            dim=1,
        )
        control_points = self.fit_control_points(
            exp_polyline
        )  # B x N_control_points x 2

        # read out the Beziers curve points at the given time parameters
        roll_out_traj = self.bezier_curve(control_points, t)  # B x T' x 2

        # compute the yaw angles for each t from the Bezier curve points
        exp_traj = torch.cat(
            (
                torch.zeros(
                    (roll_out_traj.shape[0], 1, 2), device=roll_out_traj.device
                ),
                roll_out_traj,
            ),
            dim=1,
        )
        segments = exp_traj[:, 1:] - exp_traj[:, :-1]  # B x T x 2
        traj_yaws = torch.atan2(segments[:, :, 1], segments[:, :, 0])  # B x T'

        if planning_mask is not None:
            # upsample the planning mask to match the roll-out trajectory
            planning_mask = planning_mask.repeat_interleave(
                self.temporal_upsampling, dim=1
            )  # B x T'
            valid_mask = torch.logical_and(valid_mask, planning_mask.bool())  # B x T'

        return roll_out_traj, traj_yaws, valid_mask

    def bezier_curve(self, control_points: torch.Tensor, t: torch.Tensor):
        """Compute the Bezier curve for batched set of control points and a batched set of t values.

        Args:
            control_points (torch.Tensor): Shape: B x N x 2
            t (torch.Tensor): Shape: B x T
        """
        n = control_points.shape[1]

        t_powers = torch.stack([t**i for i in range(n)], dim=-1)  # B x T x N
        one_minus_t_powers = torch.stack(
            [(1 - t) ** (n - 1 - i) for i in range(n)], dim=-1
        )  # B x T x N
        basis = self.binomial_coeffs * t_powers * one_minus_t_powers  # B x T x N
        curve = torch.sum(
            basis[:, :, :, None] * control_points[:, None, :, :], dim=2
        )  # B x T x 2
        return curve

    def fit_control_points(self, polyline: torch.Tensor) -> torch.Tensor:
        """Fit Bezier curve control points to a given polyline using least squares: AX = B.

        Args:
            polyline (torch.Tensor): Shape: B x N_fit x 2

        Returns:
            torch.Tensor: Shape: B x N_control_points x 2
        """
        bs = polyline.size(0)
        control_points_x = torch.linalg.lstsq(
            self.a_matrix.expand(bs, -1, -1), polyline[:, :, 0]
        ).solution
        control_points_y = torch.linalg.lstsq(
            self.a_matrix.expand(bs, -1, -1), polyline[:, :, 1]
        ).solution
        control_points = torch.stack([control_points_x, control_points_y], dim=-1)
        return control_points

    def _precompute_bezier_fit_lse(self, n_fit: int, n_bezier: int):
        """Pre-compute the matrix A of the linear system AX = B for the Bezier curve fitting.

        Args:
            n_fit (int): Number of points in the polyline to fit.
            n_ctrl (int): The degree of the Bezier curve.
        """

        a_matrix = torch.zeros(n_fit, n_bezier + 1, dtype=torch.float32)
        for i in range(n_fit):
            t = i / (n_fit - 1)
            for j in range(n_bezier + 1):
                a_matrix[i, j] = (
                    math.comb(self.n_bezier, j) * (t**j) * ((1 - t) ** (n_bezier - j))
                )
        self.register_buffer(
            "a_matrix", a_matrix.unsqueeze(0)
        )  # B x N_fit x (N_bezier+1)
