# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch
import math


from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from tasks.base.vec_task import VecTask


class PegInHole(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.num_props = self.cfg["env"]["numProps"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1/60.

        # prop dimensions
        self.prop_width = 0.08
        self.prop_height = 0.08
        self.prop_length = 0.08
        self.prop_spacing = 0.09

        num_obs = 23
        num_acts = 9

        # self.cfg["env"]["numObservations"] = 23
        self.cfg["env"]["numObservations"] = 21

        self.cfg["env"]["numActions"] = 9

        super().__init__(config=self.cfg, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(
            self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.z_unit_tensor = to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # create some wrapper tensors for different slices
        self.franka_default_dof_pos = to_torch(
            [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = self.dof_state.view(
            self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]
        # self.cabinet_dof_state = self.dof_state.view(
        #     self.num_envs, -1, 2)[:, self.num_franka_dofs:]
        # self.cabinet_dof_pos = self.cabinet_dof_state[..., 0]
        # self.cabinet_dof_vel = self.cabinet_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(
            rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        # self.root_state_tensor = gymtorch.wrap_tensor(
        #     actor_root_state_tensor).view(self.num_envs, -1, 13)
        self.root_state_tensor = gymtorch.wrap_tensor(
            actor_root_state_tensor).view(-1, 13)

        # if self.num_props > 0:
        #     self.prop_states = self.root_state_tensor[:, 2:]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.franka_dof_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        # self.global_indices = torch.arange(
        #     self.num_envs * (2 + self.num_props), dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.global_indices = torch.arange(
            self.num_envs * (7), dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        #contact forces
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(
            self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3)  # shape: num_envs,num_bodies, xyz axis

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
        # cabinet_asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet_2.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(
                __file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get(
                "assetFileNameFranka", franka_asset_file)
            # cabinet_asset_file = self.cfg["env"]["asset"].get(
            #     "assetFileNameCabinet", cabinet_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(
            self.sim, asset_root, franka_asset_file, asset_options)

        # # load cabinet asset
        # asset_options.flip_visual_attachments = False
        # asset_options.collapse_fixed_joints = True
        # asset_options.disable_gravity = False
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # asset_options.armature = 0.005
        # cabinet_asset = self.gym.load_asset(
        #     self.sim, asset_root, cabinet_asset_file, asset_options)

        # create table asset
        self.table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(
            self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, asset_options)

        # create hole asset
        hole_asset_height = 0.02
        hole_size = gymapi.Vec3(0.06, 0.06, hole_asset_height)
        self.external_size = gymapi.Vec3(0.2, 0.2, hole_asset_height)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        # Part A
        hole_part_a_asset = self.gym.create_box(
            self.sim, (self.external_size.x-hole_size.x)/2.0, hole_size.y, hole_asset_height, asset_options)
        # Part B
        hole_part_b_asset = self.gym.create_box(
            self.sim, self.external_size.x, (self.external_size.y-hole_size.y)/2.0, hole_asset_height, asset_options)

        # create box asset
        self.box_thickness = 0.045
        self.box_height = self.box_thickness
        asset_options = gymapi.AssetOptions()
        box_asset = self.gym.create_box(
            self.sim, self.box_thickness, self.box_thickness, self.box_height, asset_options)

        franka_dof_stiffness = to_torch(
            [400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch(
            [80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(
            franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
        # self.num_cabinet_bodies = self.gym.get_asset_rigid_body_count(
        #     cabinet_asset)
        # self.num_cabinet_dofs = self.gym.get_asset_dof_count(cabinet_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)
        # print("num cabinet bodies: ", self.num_cabinet_bodies)
        # print("num cabinet dofs: ", self.num_cabinet_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])

        self.franka_dof_lower_limits = to_torch(
            self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(
            self.franka_dof_upper_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(
            self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        # # set cabinet dof properties
        # cabinet_dof_props = self.gym.get_asset_dof_properties(cabinet_asset)
        # for i in range(self.num_cabinet_dofs):
        #     cabinet_dof_props['damping'][i] = 10.0

        # create prop assets
        # box_opts = gymapi.AssetOptions()
        # box_opts.density = 400
        # prop_asset = self.gym.create_box(
        #     self.sim, self.prop_width, self.prop_height, self.prop_width, box_opts)
        self.num_props = 0  # TODO: test
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * self.table_dims.z)

        # Hole poses
        # Part A1
        part_a_1_pose = gymapi.Transform()
        part_a_1_pose.p = gymapi.Vec3(self.table_pose.p.x-(
            hole_size.x/2.0)-(self.external_size.x-hole_size.x)/4.0, self.table_pose.p.y, self.table_dims.z+0.5*hole_asset_height)
        # part_a_1_pose.r = gymapi.Quat.from_axis_angle(
        #     gymapi.Vec3(0, 0, 1), math.pi)
        # Part A2
        part_a_2_pose = gymapi.Transform()
        part_a_2_pose.p = gymapi.Vec3(self.table_pose.p.x+(
            hole_size.x/2.0)+(self.external_size.x-hole_size.x)/4.0, self.table_pose.p.y, self.table_dims.z+0.5*hole_asset_height)
        # part_a_2_pose.r = gymapi.Quat.from_axis_angle(
        #     gymapi.Vec3(0, 0, 1), math.pi)
        # Part B1
        part_b_1_pose = gymapi.Transform()
        part_b_1_pose.p = gymapi.Vec3(self.table_pose.p.x, self.table_pose.p.y-(
            hole_size.y/2.0)-(self.external_size.y-hole_size.y)/4.0, self.table_dims.z+0.5*hole_asset_height)
        # part_b_1_pose.r = gymapi.Quat.from_axis_angle(
        #     gymapi.Vec3(0, 0, 1), math.pi)
        # Part B2
        part_b_2_pose = gymapi.Transform()
        part_b_2_pose.p = gymapi.Vec3(self.table_pose.p.x, self.table_pose.p.y+(
            hole_size.y/2.0)+(self.external_size.y-hole_size.y)/4.0, self.table_dims.z+0.5*hole_asset_height)
        # part_b_2_pose.r = gymapi.Quat.from_axis_angle(
        #     gymapi.Vec3(0, 0, 1), math.pi)

        box_pose = gymapi.Transform()

        # cabinet_start_pose = gymapi.Transform()
        # cabinet_start_pose.p = gymapi.Vec3(
        #     *get_axis_params(0.4, self.up_axis_idx))

        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * self.table_dims.z)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        # num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        # num_cabinet_shapes = self.gym.get_asset_rigid_shape_count(
        #     cabinet_asset)
        num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)

        num_box_bodies = self.gym.get_asset_rigid_body_count(box_asset)
        num_box_shapes = self.gym.get_asset_rigid_shape_count(box_asset)

        num_hole_parts_bodies = 2 * \
            self.gym.get_asset_rigid_body_count(
                hole_part_a_asset) + 2 * self.gym.get_asset_rigid_body_count(hole_part_b_asset)
        num_hole_parts_shapes = 2 * \
            self.gym.get_asset_rigid_shape_count(
                hole_part_a_asset) + 2 * self.gym.get_asset_rigid_shape_count(hole_part_b_asset)

        max_agg_bodies = num_franka_bodies + \
            num_table_bodies + num_box_bodies+num_hole_parts_bodies
        max_agg_shapes = num_franka_shapes + \
            num_table_shapes + num_box_shapes+num_hole_parts_shapes

        self.frankas = []
        # self.cabinets = []
        self.tables = []
        self.default_prop_states = []
        self.prop_start = []
        self.envs = []
        self.box_idxs = []
        self.boxes = []
        print("starting setting the environtment")

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(
                    env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add franka

            franka_actor = self.gym.create_actor(
                env_ptr, franka_asset, franka_start_pose, "franka", i, 1, 0)
            self.gym.set_actor_dof_properties(
                env_ptr, franka_actor, franka_dof_props)
            if i == 0:
                print("added franka")

            # add table

            table_handle = self.gym.create_actor(
                env_ptr, table_asset, self.table_pose, "table", i, 0)
            if i == 0:
                print("added table")

            # add hole
            part_a_1_handle = self.gym.create_actor(
                env_ptr, hole_part_a_asset, part_a_1_pose, "hole_part_a_1", i, 0)
            part_a_2_handle = self.gym.create_actor(
                env_ptr, hole_part_a_asset, part_a_2_pose, "hole_part_a_2", i, 0)
            part_b_1_handle = self.gym.create_actor(
                env_ptr, hole_part_b_asset, part_b_1_pose, "hole_part_b_1", i, 0)
            part_b_2_handle = self.gym.create_actor(
                env_ptr, hole_part_b_asset, part_b_2_pose, "hole_part_b_2", i, 0)
            if i == 0:
                print("added hole")
            # add box
            # box_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
            # box_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
            box_pose.p.x = self.table_pose.p.x + 0.5*self.external_size.x+0.25 * \
                (self.table_dims.x-self.external_size.x) + np.random.uniform(-1,
                                                                             1)*(0.25*(self.table_dims.x-self.external_size.x)-self.box_thickness)
            box_pose.p.y = self.table_pose.p.y + \
                np.random.uniform(-1, 1) * \
                (0.5*self.table_dims.y-self.box_thickness)
            box_pose.p.z = self.table_dims.z + 0.5 * self.box_height
            box_pose.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(0, 0, 1), np.random.uniform(-1, 1)*math.pi)

            box_handle = self.gym.create_actor(
                env_ptr, box_asset, box_pose, "box", i, 0)
            color = gymapi.Vec3(np.random.uniform(
                0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(
                env_ptr, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            # get global index of box in rigid body state tensor
            box_idx = self.gym.get_actor_index(
                env_ptr, box_handle, gymapi.DOMAIN_SIM)
            self.box_idxs.append(box_idx)
            if i == 0:
                print("added box")

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(
                    env_ptr, max_agg_bodies, max_agg_shapes, True)

            # cabinet_pose = cabinet_start_pose
            # cabinet_pose.p.x += self.start_position_noise * \
            #     (np.random.rand() - 0.5)
            # dz = 0.5 * np.random.rand()
            # dy = np.random.rand() - 0.5
            # cabinet_pose.p.y += self.start_position_noise * dy
            # cabinet_pose.p.z += self.start_position_noise * dz
            # cabinet_actor = self.gym.create_actor(
            #     env_ptr, cabinet_asset, cabinet_pose, "cabinet", i, 2, 0)
            # self.gym.set_actor_dof_properties(
            #     env_ptr, cabinet_actor, cabinet_dof_props)
            # add table

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(
                    env_ptr, max_agg_bodies, max_agg_shapes, True)

            # if self.num_props > 0:
            #     self.prop_start.append(self.gym.get_sim_actor_count(self.sim))
            #     drawer_handle = self.gym.find_actor_rigid_body_handle(
            #         env_ptr, cabinet_actor, "drawer_top")
            #     drawer_pose = self.gym.get_rigid_transform(
            #         env_ptr, drawer_handle)

            #     props_per_row = int(np.ceil(np.sqrt(self.num_props)))
            #     xmin = -0.5 * self.prop_spacing * (props_per_row - 1)
            #     yzmin = -0.5 * self.prop_spacing * (props_per_row - 1)

            #     prop_count = 0
            #     for j in range(props_per_row):
            #         prop_up = yzmin + j * self.prop_spacing
            #         for k in range(props_per_row):
            #             if prop_count >= self.num_props:
            #                 break
            #             propx = xmin + k * self.prop_spacing
            #             prop_state_pose = gymapi.Transform()
            #             prop_state_pose.p.x = drawer_pose.p.x + propx
            #             propz, propy = 0, prop_up
            #             prop_state_pose.p.y = drawer_pose.p.y + propy
            #             prop_state_pose.p.z = drawer_pose.p.z + propz
            #             prop_state_pose.r = gymapi.Quat(0, 0, 0, 1)
            #             prop_handle = self.gym.create_actor(
            #                 env_ptr, prop_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
            #             prop_count += 1

            #             prop_idx = j * props_per_row + k
            #             self.default_prop_states.append([prop_state_pose.p.x, prop_state_pose.p.y, prop_state_pose.p.z,
            #                                              prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z, prop_state_pose.r.w,
            #                                              0, 0, 0, 0, 0, 0])
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.boxes.append(box_handle)
            # self.cabinets.append(cabinet_actor)
            # self.tables.append(table_handle)
        print("environtment set")
        self.hand_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, franka_actor, "panda_link7")
        # self.hand_handle = self.gym.find_actor_rigid_body_handle(
        #     env_ptr, franka_actor, "panda_hand")
        # self.drawer_handle = self.gym.find_actor_rigid_body_handle(
        #     env_ptr, cabinet_actor, "drawer_top")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, franka_actor, "panda_leftfinger")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, franka_actor, "panda_rightfinger")
        # self.default_prop_states = to_torch(
        #     self.default_prop_states, device=self.device, dtype=torch.float).view(self.num_envs, self.num_props, 13)
        self.box_idxs = to_torch(
            self.box_idxs, dtype=torch.long, device=self.device).view(self.num_envs, 1, 1)
        #Index for evaluate contact forces
        self.base_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.frankas[0], "panda_link4")
        print("BASE INDEX:",self.base_index)

        print("initializing data")
        self.init_data()
        print("Data initialized")

    def init_data(self):
        hand = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.frankas[0], "panda_link7")
        # hand = self.gym.find_actor_rigid_body_handle(
        #     self.envs[0], self.frankas[0], "panda_hand")
        lfinger = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.frankas[0], "panda_leftfinger")
        rfinger = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.frankas[0], "panda_rightfinger")

        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
        rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)

        finger_pose = gymapi.Transform()
        finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
        finger_pose.r = lfinger_pose.r

        hand_pose_inv = hand_pose.inverse()
        # grasp_pose_axis = 1
        grasp_pose_axis = 2

        franka_local_grasp_pose = hand_pose_inv * finger_pose
        # franka_local_grasp_pose.p += gymapi.Vec3(
        #     *get_axis_params(0.04, grasp_pose_axis))
        franka_local_grasp_pose.p += gymapi.Vec3(
            *get_axis_params(0.045, grasp_pose_axis))
        # print("Axis offset: ", *get_axis_params(0.045, grasp_pose_axis))
        self.franka_local_grasp_pos = to_torch([franka_local_grasp_pose.p.x, franka_local_grasp_pose.p.y,
                                                franka_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.franka_local_grasp_rot = to_torch([franka_local_grasp_pose.r.x, franka_local_grasp_pose.r.y,
                                                franka_local_grasp_pose.r.z, franka_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        # drawer_local_grasp_pose = gymapi.Transform()
        # drawer_local_grasp_pose.p = gymapi.Vec3(
        #     *get_axis_params(0.01, grasp_pose_axis, 0.3))
        # drawer_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        # self.drawer_local_grasp_pos = to_torch([drawer_local_grasp_pose.p.x, drawer_local_grasp_pose.p.y,
        #                                         drawer_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        # self.drawer_local_grasp_rot = to_torch([drawer_local_grasp_pose.r.x, drawer_local_grasp_pose.r.y,
        #                                         drawer_local_grasp_pose.r.z, drawer_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        self.gripper_forward_axis = to_torch(
            [0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        # self.drawer_inward_axis = to_torch(
        #     [-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis = to_torch(
            [0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        # self.drawer_up_axis = to_torch(
        #     [0, 0, 1], device=self.device).repeat((self.num_envs, 1))

        self.franka_grasp_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_grasp_rot = torch.zeros_like(self.franka_local_grasp_rot)
        self.franka_grasp_rot[..., -1] = 1  # xyzw
        # self.drawer_grasp_pos = torch.zeros_like(self.drawer_local_grasp_pos)
        # self.drawer_grasp_rot = torch.zeros_like(self.drawer_local_grasp_rot)
        # self.drawer_grasp_rot[..., -1] = 1
        self.franka_lfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_rfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_lfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)
        self.franka_rfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)

    def compute_reward(self, actions):

        # self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
        #     self.reset_buf, self.progress_buf, self.actions, self.cabinet_dof_pos,
        #     self.franka_grasp_pos, self.drawer_grasp_pos, self.franka_grasp_rot, self.drawer_grasp_rot,
        #     self.franka_lfinger_pos, self.franka_rfinger_pos,
        #     self.gripper_forward_axis, self.drawer_inward_axis, self.gripper_up_axis, self.drawer_up_axis,
        #     self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale, self.open_reward_scale,
        #     self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset, self.max_episode_length
        # )

        table_pos = to_torch(
            [self.table_pose.p.x, self.table_pose.p.y, self.table_pose.p.z], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions,
            self.hand_pos,
            self.franka_grasp_pos,  self.franka_grasp_rot,
            self.franka_lfinger_pos, self.franka_rfinger_pos,
            self.object_pos, table_pos,
            self.contact_forces, torch.range(0,8,device=self.device,dtype=torch.long,requires_grad=False),
            self.gripper_forward_axis,  self.gripper_up_axis,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale, self.open_reward_scale,
            self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset, self.max_episode_length
        )

    def compute_observations(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)


        self.hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        self.hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        # print("root state tensor shape is: ", self.root_state_tensor.size())

        self.object_pose = self.root_state_tensor[self.box_idxs, 0:3]
        self.object_pos = self.root_state_tensor[self.box_idxs,
                                                 0:3][:, 0, 0, :]
        self.object_rot = self.root_state_tensor[self.box_idxs,
                                                 3:7][:, 0, 0, :]
        self.object_linvel = self.root_state_tensor[self.box_idxs,
                                                    7:10][:, 0, 0, :]
        self.object_angvel = self.root_state_tensor[self.box_idxs,
                                                    10:13][:, 0, 0, :]
        # drawer_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        # drawer_rot = self.rigid_body_states[:, self.drawer_handle][:, 3:7]

        # self.franka_grasp_rot[:], self.franka_grasp_pos[:], self.drawer_grasp_rot[:], self.drawer_grasp_pos[:] = \
        #     compute_grasp_transforms(hand_rot, hand_pos, self.franka_local_grasp_rot, self.franka_local_grasp_pos,
        #                              drawer_rot, drawer_pos, self.drawer_local_grasp_rot, self.drawer_local_grasp_pos
        #                              )

        self.franka_grasp_rot[:], self.franka_grasp_pos[:] = compute_grasp_transforms(
            self.hand_rot, self.hand_pos, self.franka_local_grasp_rot, self.franka_local_grasp_pos)
        # self.franka_grasp_rot[:] = hand_rot
        # self.franka_grasp_pos[:] = hand_pos

        self.franka_lfinger_pos = self.rigid_body_states[:,
                                                         self.lfinger_handle][:, 0:3]
        self.franka_rfinger_pos = self.rigid_body_states[:,
                                                         self.rfinger_handle][:, 0:3]
        self.franka_lfinger_rot = self.rigid_body_states[:,
                                                         self.lfinger_handle][:, 3:7]
        self.franka_rfinger_rot = self.rigid_body_states[:,
                                                         self.rfinger_handle][:, 3:7]

        dof_pos_scaled = (2.0 * (self.franka_dof_pos - self.franka_dof_lower_limits)
                          / (self.franka_dof_upper_limits - self.franka_dof_lower_limits) - 1.0)
        # to_target = self.drawer_grasp_pos - self.franka_grasp_pos
        # self.obs_buf = torch.cat((dof_pos_scaled, self.franka_dof_vel * self.dof_vel_scale, to_target,
        #                           self.cabinet_dof_pos[:, 3].unsqueeze(-1), self.cabinet_dof_vel[:, 3].unsqueeze(-1)), dim=-1)
        to_target = self.object_pos - self.franka_grasp_pos
        self.obs_buf = torch.cat(
            (dof_pos_scaled, self.franka_dof_vel * self.dof_vel_scale, to_target), dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(
                0) + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        self.franka_dof_pos[env_ids, :] = pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(
            self.franka_dof_vel[env_ids])
        self.franka_dof_targets[env_ids, :self.num_franka_dofs] = pos

        # # reset cabinet
        # self.cabinet_dof_state[env_ids, :] = torch.zeros_like(
        #     self.cabinet_dof_state[env_ids])

        # # reset props
        # if self.num_props > 0:
        #     prop_indices = self.global_indices[env_ids, 2:].flatten()
        #     self.prop_states[env_ids] = self.default_prop_states[env_ids]
        #     self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                                  gymtorch.unwrap_tensor(
        #                                                      self.root_state_tensor),
        #                                                  gymtorch.unwrap_tensor(prop_indices), len(prop_indices))

        # Reset object
        # generate random values

        # rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids),
        #                                3, 1), device=self.device)

        # reset object
        # print("state is: ",
        #       (self.root_state_tensor[self.box_idxs[env_ids], 0][:,0,0]).size())
        # self.root_state_tensor[self.box_idxs[env_ids], 0][:, 0, 0] = torch.full((len(env_ids),), self.table_pose.p.x + 0.5*self.external_size.x+0.25 *
        #     (self.table_dims.x-self.external_size.x),device=self.device) + rand_floats[:,0]*(0.25*(self.table_dims.x-self.external_size.x)-self.box_thickness)

        self.root_state_tensor[self.box_idxs[env_ids], 0] = torch.full_like(self.root_state_tensor[self.box_idxs[env_ids], 0], self.table_pose.p.x + 0.5*self.external_size.x+0.25 * (
            self.table_dims.x-self.external_size.x)) + (1 - 2 *
                                                        torch.rand_like(self.root_state_tensor[self.box_idxs[env_ids], 0]))*(0.25*(self.table_dims.x-self.external_size.x)-self.box_thickness)

        self.root_state_tensor[self.box_idxs[env_ids], 1] = torch.full_like(
            self.root_state_tensor[self.box_idxs[env_ids], 1], self.table_pose.p.y) + (1 - 2 *
                                                                                       torch.rand_like(self.root_state_tensor[self.box_idxs[env_ids], 0]))*(0.5*self.table_dims.y-self.box_thickness)
        self.root_state_tensor[self.box_idxs[env_ids], 2] = torch.full_like(
            self.root_state_tensor[self.box_idxs[env_ids], 2], self.table_dims.z + 0.5 * self.box_height)

        self.root_state_tensor[self.box_idxs[env_ids], 3:7] = quat_from_euler_xyz(
            torch.zeros((len(env_ids), 1, 1), device=self.device), torch.zeros((len(env_ids), 1, 1), device=self.device), torch.rand((len(env_ids), 1, 1), device=self.device)*2*np.pi)

        object_indices = self.global_indices[env_ids, -1].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(
                                                         self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        multi_env_ids_int32 = self.global_indices[env_ids, :1].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(
                                                            self.franka_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(
                                                  self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):

        self.actions = actions.clone().to(self.device)
        targets = self.franka_dof_targets[:, :self.num_franka_dofs] + \
            self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(
            self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.franka_dof_targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                px = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch(
                    [1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch(
                    [0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch(
                    [0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [
                                   p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [
                                   p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [
                                   p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.hand_pos[i] + quat_apply(self.hand_rot[i], to_torch(
                    [1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.hand_pos[i] + quat_apply(self.hand_rot[i], to_torch(
                    [0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.hand_pos[i] + quat_apply(self.hand_rot[i], to_torch(
                    [0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.hand_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [
                                   p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [
                                   p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [
                                   p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                # p0 = self.franka_grasp_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [
                #                    p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [
                #                    p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [
                #                    p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                px = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch(
                    [1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch(
                    [0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch(
                    [0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [
                                   p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [
                                   p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [
                                   p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

        #         px = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch(
        #             [1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        #         py = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch(
        #             [0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        #         pz = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch(
        #             [0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        #         p0 = self.drawer_grasp_pos[i].cpu().numpy()
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [
        #                            p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [
        #                            p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [
        #                            p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch(
                    [1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch(
                    [0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch(
                    [0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_lfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [
                                   p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [
                                   p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [
                                   p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch(
                    [1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch(
                    [0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch(
                    [0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_rfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [
                                   p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [
                                   p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [
                                   p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

#####################################################################
###=========================jit functions=========================###
#####################################################################
# def compute_franka_reward(
#     reset_buf, progress_buf, actions, cabinet_dof_pos,
#     franka_grasp_pos, drawer_grasp_pos, franka_grasp_rot, drawer_grasp_rot,
#     franka_lfinger_pos, franka_rfinger_pos,
#     gripper_forward_axis, drawer_inward_axis, gripper_up_axis, drawer_up_axis,
#     num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
#     finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
# ):
#     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]


@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions,
    franka_hand_pos,
    franka_grasp_pos,  franka_grasp_rot,
    franka_lfinger_pos, franka_rfinger_pos,
    object_pos, hole_pos,
    contact_forces, force_indices,
    gripper_forward_axis,  gripper_up_axis,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
):
    # type: (Tensor,Tensor, Tensor, Tensor,Tensor, Tensor, Tensor,Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

    # distance vertical hand
    d_hand = torch.norm(franka_grasp_pos[:,0:1] - franka_hand_pos[:,0:1], p=2, dim=-1)
    dist_hand_reward = 1.0 / (1.0 + d_hand ** 2)
    dist_hand_reward *= dist_hand_reward
    dist_hand_reward = torch.where(
        d_hand <= 0.045, dist_hand_reward * 2, dist_hand_reward)

    # distance from left finger to the object
    d_lfinger = torch.norm(franka_lfinger_pos - object_pos, p=2, dim=-1)
    dist_lfinger_reward = 1.0 / (1.0 + d_lfinger ** 2)
    dist_lfinger_reward *= dist_lfinger_reward
    dist_lfinger_reward *= dist_lfinger_reward
    dist_lfinger_reward = torch.where(
        d_lfinger <= 0.045, dist_lfinger_reward * 2, dist_lfinger_reward)

    # distance from right finger to the object
    d_rfinger = torch.norm(franka_rfinger_pos - object_pos, p=2, dim=-1)
    dist_rfinger_reward = 1.0 / (1.0 + d_rfinger ** 2)
    dist_rfinger_reward *= dist_rfinger_reward
    dist_rfinger_reward *= dist_rfinger_reward
    dist_rfinger_reward = torch.where(
        d_rfinger <= 0.045, dist_rfinger_reward * 2, dist_rfinger_reward)

    # distance from object to hole
    d_hole = torch.norm(hole_pos[:, 0:1] - object_pos[:, 0:1], p=2, dim=-1)
    dist_hole_reward = 1.0 / (1.0 + d_hole ** 2)
    dist_hole_reward *= dist_hole_reward
    dist_hole_reward *= dist_hole_reward
    dist_hole_reward = torch.where(
        d_hole <= 0.045, dist_hole_reward * 2, dist_hole_reward)

    # # distance from hand to the drawer
    # d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    # dist_reward = 1.0 / (1.0 + d ** 2)
    # dist_reward *= dist_reward
    # dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    # axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
    # axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
    # axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
    # axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

    # dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(
    #     num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
    # dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(
    #     num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of up axis for gripper
    # # reward for matching the orientation of the hand to the drawer (fingers wrapped)
    # rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 +
    #                     torch.sign(dot2) * dot2 ** 2)

    # # bonus if left finger is above the drawer handle and right below
    # around_handle_reward = torch.zeros_like(rot_reward)
    # around_handle_reward = torch.where(franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
    #                                    torch.where(franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
    #                                                around_handle_reward + 0.5, around_handle_reward), around_handle_reward)
    # # reward for distance of each finger from the drawer
    # finger_dist_reward = torch.zeros_like(rot_reward)
    # lfinger_dist = torch.abs(franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
    # rfinger_dist = torch.abs(franka_rfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
    # finger_dist_reward = torch.where(franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
    #                                  torch.where(franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
    #                                              (0.04 - lfinger_dist) + (0.04 - rfinger_dist), finger_dist_reward), finger_dist_reward)

    # bonus if hand is above the box
    above_reward = torch.zeros_like(dist_hole_reward)
    above_reward = torch.where(
        franka_hand_pos[:, 2] > 0.05+object_pos[:, 2], above_reward+0.1, above_reward)

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # # how far the cabinet has been opened out
    # # drawer_top_joint
    # open_reward = cabinet_dof_pos[:, 3] * \
    #     around_handle_reward + cabinet_dof_pos[:, 3]

    # rewards = dist_reward_scale * dist_reward + rot_reward_scale * rot_reward \
    #     + around_handle_reward_scale * around_handle_reward + open_reward_scale * open_reward \
    #     + finger_dist_reward_scale * finger_dist_reward - \
    #     action_penalty_scale * action_penalty

    # Existance penalty
    existance_penalty = progress_buf/max_episode_length

    rewards = (dist_reward_scale*dist_lfinger_reward + dist_reward_scale*dist_rfinger_reward) + \
        2*dist_reward_scale*dist_hole_reward - action_penalty_scale * \
        action_penalty
    # rewards = dist_hand_reward - action_penalty_scale * \
    #     action_penalty

    # # bonus for opening drawer properly
    # rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.5, rewards)
    # rewards = torch.where(
    #     cabinet_dof_pos[:, 3] > 0.2, rewards + around_handle_reward, rewards)
    # rewards = torch.where(
    #     cabinet_dof_pos[:, 3] > 0.39, rewards + (2.0 * around_handle_reward), rewards)

    # # prevent bad style in opening drawer
    # rewards = torch.where(franka_lfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
    #                       torch.ones_like(rewards) * -1, rewards)
    # rewards = torch.where(franka_rfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
    #                       torch.ones_like(rewards) * -1, rewards)

    # # reset if drawer is open or max length reached
    # reset_buf = torch.where(
    #     cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    reset_buf = torch.where(
        object_pos[:, 2] < 0.10, torch.ones_like(reset_buf), reset_buf)

    reset_buf = reset_buf | torch.any(torch.norm(
        contact_forces[:, force_indices, :], dim=2) > 1., dim=1)

    return rewards, reset_buf


@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
                             ):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    global_franka_rot, global_franka_pos = tf_combine(
        hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos)

    return global_franka_rot, global_franka_pos

# def compute_grasp_transforms(hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos,
#                                 drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
#                                 ):
#     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

# global_franka_rot, global_franka_pos = tf_combine(
#     hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos)
# global_drawer_rot, global_drawer_pos = tf_combine(
#     drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos)

# return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos
