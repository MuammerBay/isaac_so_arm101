# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Central configuration for PicknPlace task positions and parameters.
This file contains all position configurations to ensure consistency across all scripts.
"""

from typing import List

class PicknPlaceConfig:
    """Central configuration for PicknPlace task"""
    
    # Robot configuration
    ROBOT_POS = [0.0, 0.0, 0.0]
    
    # Table configuration
    TABLE_POS = [0.5, 0.0, 0.0]
    
    # Target platform configuration
    PLATFORM_POS = [0.25, -0.15, 0.032]
    PLATFORM_SIZE = [0.08, 0.08, 0.02]
    
    # Target area configuration
    TARGET_AREA_POS = [0.25, -0.15, 0.043]
    TARGET_AREA_SIZE = [0.05, 0.05, 0.001]
    
    # Pickup object configuration
    CUBE_POS = [0.2, 0.0, 0.015]
    CUBE_SIZE = [0.03, 0.03, 0.03]
    
    # Goal tolerances
    PLACEMENT_TOLERANCE_XY = 0.02
    PLACEMENT_TOLERANCE_Z = 0.015
    TERMINATION_TOLERANCE_XY = 0.015
    TERMINATION_TOLERANCE_Z = 0.01
    
    # Reward parameters
    TRANSPORT_STD = 0.05
    MINIMAL_LIFT_HEIGHT = 0.04
    
    @classmethod
    def get_platform_position(cls) -> List[float]:
        """Get platform position"""
        return cls.PLATFORM_POS.copy()
    
    @classmethod  
    def get_target_area_position(cls) -> List[float]:
        """Get target area position"""
        return cls.TARGET_AREA_POS.copy()
    
    @classmethod
    def get_cube_position(cls) -> List[float]:
        """Get cube position"""
        return cls.CUBE_POS.copy()
    
    @classmethod
    def get_table_position(cls) -> List[float]:
        """Get table position"""
        return cls.TABLE_POS.copy()
    
    @classmethod
    def is_object_in_target_area(cls, obj_pos: List[float]) -> bool:
        """Check if object is on target area"""
        target = cls.TARGET_AREA_POS
        tolerance_xy = cls.PLACEMENT_TOLERANCE_XY
        tolerance_z = cls.PLACEMENT_TOLERANCE_Z
        
        # Check if object is within target area bounds
        x_in = abs(obj_pos[0] - target[0]) <= tolerance_xy
        y_in = abs(obj_pos[1] - target[1]) <= tolerance_xy  
        z_in = abs(obj_pos[2] - target[2]) <= tolerance_z
        
        return x_in and y_in and z_in
    
    @classmethod
    def get_distance_to_target(cls, obj_pos: List[float]) -> float:
        """Get distance from object to target area center"""
        import math
        target = cls.TARGET_AREA_POS
        return math.sqrt(
            (obj_pos[0] - target[0])**2 + 
            (obj_pos[1] - target[1])**2 + 
            (obj_pos[2] - target[2])**2
        )

# Create global instance
PICKNPLACE_CFG = PicknPlaceConfig()