#!/usr/bin/env python3
"""
Koordinat sistemi analizi - masa rotasyonu etkisi
"""
import numpy as np

def analyze_coordinate_system():
    print("="*70)
    print("KOORDİNAT SİSTEMİ ANALİZİ")
    print("="*70)
    
    # Mevcut konumlar
    robot_pos = np.array([0.0, 0.0, 0.0])
    table_pos = np.array([0.5, 0.0, 0.0])
    platform_pos = np.array([0.25, -0.15, 0.032])
    cube1_pos = np.array([0.12, 0.05, 0.015])
    
    # Command targets
    command_target = np.array([-0.15, -0.25, 0.043])
    reward_target = np.array([0.25, -0.15, 0.0575])
    
    print("WORLD FRAME POSİSYONLARI:")
    print(f"Robot:     {robot_pos}")
    print(f"Table:     {table_pos}")
    print(f"Platform:  {platform_pos}")
    print(f"Cube1:     {cube1_pos}")
    print()
    print("HEDEF POSİSYONLARI:")
    print(f"Command:   {command_target}")
    print(f"Reward:    {reward_target}")
    
    print("\n" + "="*70)
    print("MASA ROTASYON ETKİSİ")
    print("="*70)
    
    # Masa rotasyonu: [0.707, 0, 0, 0.707] = 90° Z ekseni etrafında
    print("Masa quaternion: [0.707, 0, 0, 0.707]")
    print("Bu 90° Z ekseni etrafında rotation demek!")
    
    # 90° rotation matrix (Z ekseni etrafında)
    rotation_90 = np.array([
        [0, -1, 0],
        [1,  0, 0], 
        [0,  0, 1]
    ])
    
    print(f"\n90° rotation matrix (Z ekseni):")
    print(rotation_90)
    
    print("\n" + "="*70)
    print("COMMAND TARGET ANALİZİ")
    print("="*70)
    
    # Command target robot frame'de
    print("Command target robot frame'de: (-0.15, -0.25)")
    print("Bu robot'un:")
    print("  X: -0.15 (geriye/arkaya)")
    print("  Y: -0.25 (sola)")
    
    # Ama masa 90° döndürülmüş!
    print("\nAMA MASA 90° DÖNDÜRÜLMÜŞ!")
    print("Eğer masa coordinate frame etkiliyorsa:")
    
    # Command target'ı masa frame'inde görelim
    command_in_table_frame = rotation_90 @ command_target[:3]
    print(f"Command masa frame'de: {command_in_table_frame}")
    
    print("\n" + "="*70)
    print("OLASI PROBLEM")
    print("="*70)
    
    print("1. Robot command'ı robot frame'de alıyor: (-0.15, -0.25)")
    print("2. Ama masa 90° döndürülmüş")
    print("3. Platform masanın üstünde")
    print("4. Coordinate transformation hatası olabilir!")
    
    print(f"\nEğer masa rotation etkili ise:")
    print(f"Command (-0.15, -0.25) → masa frame'de {command_in_table_frame[:2]}")
    
    print(f"\nReward target (world frame): {reward_target[:2]}")
    print(f"Platform (world frame): {platform_pos[:2]}")
    
    # Mesafeler
    robot_to_command_world = np.linalg.norm(robot_pos[:2] - command_target[:2])
    robot_to_reward = np.linalg.norm(robot_pos[:2] - reward_target[:2])
    robot_to_platform = np.linalg.norm(robot_pos[:2] - platform_pos[:2])
    
    print(f"\nMESAFELER (robot'tan):")
    print(f"Command target'a:  {robot_to_command_world:.3f}m")
    print(f"Reward target'a:   {robot_to_reward:.3f}m") 
    print(f"Platform'a:        {robot_to_platform:.3f}m")
    
    print(f"\n" + "="*70)
    print("SONUÇ")
    print("="*70)
    print("Robot command frame ile world frame arasında")
    print("coordinate transformation problemi olabilir.")
    print("Masa rotation'ı bu karışıklığa sebep oluyor olabilir.")

if __name__ == "__main__":
    analyze_coordinate_system()