#!/usr/bin/env python3
"""
Test script to demonstrate the two coordinate conversion modes.
"""

import numpy as np

# Mock era_world_bounds for testing
era_world_bounds = [[-0.3, 0.3], [-0.3, 0.3], [0.0, 0.4]]

def era_world_to_discrete(points_era_world):
    """Discrete mode: round to integers [0, 100]"""
    discrete_coords = []
    for point in points_era_world:
        discrete_point = []
        for i, val in enumerate(point):
            min_val, max_val = era_world_bounds[i]
            discrete_val = int(round((val - min_val) / (max_val - min_val) * 100))
            discrete_val = np.clip(discrete_val, 0, 100)
            discrete_point.append(discrete_val)
        discrete_coords.append(discrete_point)
    return discrete_coords

def era_world_to_scaled(points_era_world):
    """Scaled mode: keep as floats [0, 100]"""
    scaled_coords = []
    for point in points_era_world:
        scaled_point = []
        for i, val in enumerate(point):
            min_val, max_val = era_world_bounds[i]
            scaled_val = (val - min_val) / (max_val - min_val) * 100
            scaled_val = np.clip(scaled_val, 0.0, 100.0)
            scaled_point.append(round(scaled_val, 2))  # Round to 2 decimals for display
        scaled_coords.append(scaled_point)
    return scaled_coords

def discrete_to_era_world(discrete_coords):
    """Convert back to era_world (works for both modes)"""
    continuous_point = []
    for i, discrete_val in enumerate(discrete_coords[:3]):
        min_val, max_val = era_world_bounds[i]
        continuous_val = min_val + (discrete_val / 100.0) * (max_val - min_val)
        continuous_point.append(continuous_val)
    return np.array(continuous_point)

def test_coordinate_modes():
    """Test both coordinate modes with example data"""
    
    # Example object positions in era_world frame (meters)
    points_era_world = [
        np.array([0.15, -0.22, 0.08]),   # Object 1
        np.array([-0.10, 0.05, 0.12]),   # Object 2
        np.array([0.20, 0.15, 0.10]),    # Object 3
    ]
    
    print("=" * 80)
    print("COORDINATE MODE COMPARISON TEST")
    print("=" * 80)
    print(f"\nERA World Bounds: {era_world_bounds}")
    print(f"  X: [{era_world_bounds[0][0]}, {era_world_bounds[0][1]}] meters")
    print(f"  Y: [{era_world_bounds[1][0]}, {era_world_bounds[1][1]}] meters")
    print(f"  Z: [{era_world_bounds[2][0]}, {era_world_bounds[2][1]}] meters")
    
    print("\n" + "-" * 80)
    print("MODE 1: DISCRETE (Integer Coordinates)")
    print("-" * 80)
    
    discrete_coords = era_world_to_discrete(points_era_world)
    
    print("\nOriginal → Discrete → Reconstructed:")
    for i, (orig, disc) in enumerate(zip(points_era_world, discrete_coords), 1):
        reconstructed = discrete_to_era_world(disc)
        error = np.linalg.norm(orig - reconstructed) * 1000  # mm
        
        print(f"\nObject {i}:")
        print(f"  Original:      [{orig[0]:7.4f}, {orig[1]:7.4f}, {orig[2]:7.4f}] m")
        print(f"  Discrete:      [{disc[0]:3d}, {disc[1]:3d}, {disc[2]:3d}]")
        print(f"  Reconstructed: [{reconstructed[0]:7.4f}, {reconstructed[1]:7.4f}, {reconstructed[2]:7.4f}] m")
        print(f"  Error:         {error:.2f} mm")
    
    print("\n" + "-" * 80)
    print("MODE 2: SCALED (Float Coordinates)")
    print("-" * 80)
    
    scaled_coords = era_world_to_scaled(points_era_world)
    
    print("\nOriginal → Scaled → Reconstructed:")
    for i, (orig, scaled) in enumerate(zip(points_era_world, scaled_coords), 1):
        reconstructed = discrete_to_era_world(scaled)
        error = np.linalg.norm(orig - reconstructed) * 1000  # mm
        
        print(f"\nObject {i}:")
        print(f"  Original:      [{orig[0]:7.4f}, {orig[1]:7.4f}, {orig[2]:7.4f}] m")
        print(f"  Scaled:        [{scaled[0]:6.2f}, {scaled[1]:6.2f}, {scaled[2]:6.2f}]")
        print(f"  Reconstructed: [{reconstructed[0]:7.4f}, {reconstructed[1]:7.4f}, {reconstructed[2]:7.4f}] m")
        print(f"  Error:         {error:.2f} mm")
    
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    # Calculate average errors
    discrete_errors = []
    scaled_errors = []
    
    for orig, disc, scaled in zip(points_era_world, discrete_coords, scaled_coords):
        disc_recon = discrete_to_era_world(disc)
        scaled_recon = discrete_to_era_world(scaled)
        
        discrete_errors.append(np.linalg.norm(orig - disc_recon) * 1000)
        scaled_errors.append(np.linalg.norm(orig - scaled_recon) * 1000)
    
    print(f"\nAverage Reconstruction Error:")
    print(f"  Discrete Mode: {np.mean(discrete_errors):.2f} mm (max: {np.max(discrete_errors):.2f} mm)")
    print(f"  Scaled Mode:   {np.mean(scaled_errors):.2f} mm (max: {np.max(scaled_errors):.2f} mm)")
    
    print(f"\nPrecision Improvement:")
    improvement = (np.mean(discrete_errors) - np.mean(scaled_errors)) / np.mean(discrete_errors) * 100
    print(f"  {improvement:.1f}% better with scaled mode")
    
    print("\n" + "=" * 80)
    print("EXAMPLE PROMPTS SENT TO MODEL")
    print("=" * 80)
    
    print("\nDiscrete Mode:")
    discrete_info = {f"'object {i+1}'": coord for i, coord in enumerate(discrete_coords)}
    print(f"  additional_info: {discrete_info}")
    
    print("\nScaled Mode:")
    scaled_info = {f"'object {i+1}'": coord for i, coord in enumerate(scaled_coords)}
    print(f"  additional_info: {scaled_info}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_coordinate_modes()


