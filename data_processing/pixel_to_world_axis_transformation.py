# Reference point in the simulation coordinates
sim_ref_x, sim_ref_y = 0, 0  # Origin
# Reference point in the pixel coordinates
pixel_ref_x, pixel_ref_y = 185, 270

# Scaling factors (assuming 1 meter = 1 pixel for demonstration purposes)
# You will need to adjust these according to your actual simulation scale
scale_x = 1
scale_y = 1

def simulation_to_pixel(sim_x, sim_y):
    # Apply scaling factors
    scaled_x = sim_x * scale_x
    scaled_y = sim_y * scale_y
    
    # Translate simulation coordinates to pixel coordinates using the reference point
    pixel_x = pixel_ref_x + scaled_x
    # Invert the y-axis and apply the translation
    pixel_y = pixel_ref_y - scaled_y
    
    return pixel_x, pixel_y

# Example usage:
# Convert simulation coordinates (10, -5) to pixel coordinates
pixel_coords = simulation_to_pixel(18, 2)
print(f"Pixel coordinates: {pixel_coords}")
