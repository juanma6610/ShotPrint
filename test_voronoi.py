
import os
import numpy as np
import pickle
from game import Game

def test_voronoi_sum():
    # Load the first game we find
    game = Game('01.13.2016', 'GSW', 'DEN')


    physics = game.get_player_physics(500)
    
    print(f"Frame {500} Physics:")
    for i in range(1, 4): # Check first few players
        p = physics[i]
        print(f"Player {i}: Speed={p['speed']:.2f} ft/s, Accel={p['accel_mag']:.2f} ft/s^2")
    
    # Test a frame (e.g. frame 1000)
    frame = 1000
    home_vor, away_vor = game.get_voronoi_areas(frame)
    # Test with distance-based control
    game.plot_frame(frame, show_spacing="vor", show_spacing_team="home", 
                    show_velocity=True, show_control='home')
    
    # Test with physics-based time-to-reach control (more realistic)
    # game.plot_frame(frame, show_spacing="vor", show_spacing_team="home", 
    #                 show_velocity=True, show_control='home', use_time_control=True)
    
    # Uncomment to generate animation with all features
    # game.animate_play(frame, 10, show_spacing="vor", show_spacing_team="home", 
    #                  show_velocity=True, show_control='home', use_time_control=True)
    total_area = home_vor + away_vor
    
    print(f"Frame {frame} Voronoi Areas:")
    print(f"Home: {home_vor:.2f}")
    print(f"Away: {away_vor:.2f}")
    print(f"Total: {total_area:.2f}")
    print(f"Expected: 4700.00")
    
    # If the total is close to 4700, it's working well
    if abs(total_area - 4700) < 1:
        print("Voronoi areas sum to court size. SUCCESS.")
    else:
        print("Voronoi areas DO NOT sum to court size. Investigation needed.")

if __name__ == "__main__":
    if not os.path.exists('data'):
        print("Data directory not found. Cannot run test.")
    else:
        test_voronoi_sum()




