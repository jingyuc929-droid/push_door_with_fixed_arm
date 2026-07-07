import isaaclab.envs.mdp as mdp_std

from .mit_actions import ARX5MITJointActionCfg

# -------- rewards --------
from .rewards import (
    # helpers
    _body_pose_w,
    fingertip_mid_to_handle_distance,

    # shaping
    approach_handle_inv_square,
    align_grasp_around_handle_local,
    align_grasp_pose_v2,
    close_gripper_shaping_when_ready,

    # grasp + success
    grasp_handle_reward,
    grasp_handle_reward_preunlock_only,
    grasp_success_bonus,
    compute_stage1_grasp_quality,
    grasp_quality_keep_reward,
    anti_release_after_press_to_open,

    #unlock progress
    press_handle_after_grasp_vel,
    stall_penalty_after_grasp,
    stall_penalty_after_grasp_pos,
    near_unlock_stall_penalty,
    unlock_handle_progress_mixed,
    unlock_success_bonus,
    physical_unlock_transition_bonus,
    release_after_unlock_failure,

    # push door progress after unlock
    push_door_progress_after_unlock,
    push_door_progress_after_unlock_success_only,
    stage_gated_door_reward,
)

# -------- terminations --------
from .terminations import (
    # keep these if you still use them anywhere
    sustained_contact,
    sustained_two_sensors_contact,

    # new “grasp is achieved” termination (handle-only filtered + near + wrap + closedness)
    grasp_handle_sustained,

    # new "unlock after grasp" termination (handle-only filtered + near + wrap + closedness + door unlocked)
    unlock_handle_after_grasp,

    # new "push door after unlock" termination (door-only filtered +  + wrap + unlocked
    door_open_success_only,
    release_after_grasp_failure,

)
# -------- stage ---------------
from .stage import (
    staged_reset_from_archive,
)

# -------- observations --------
from .observations import (
    ee_pos_in_handle_frame,
    ee_quat_error_handle_frame,
    gripper_opening,
    ee_tcp_pose_w,
    ee_pos_in_noisy_handle_frame,
    ee_quat_error_noisy_handle_frame,
    finger_contact_norms,
    gripper_width,
    last_action,
    last_applied_arm_delta,
    arm_q_des_error,
)

# -------- events --------
from .events import (
    update_door_lock_hysteresis_delayed_release,
)
