def txt_to_ply(input_file, output_file):
    with open(input_file, 'r') as txt_file:
        line = txt_file.readline().strip()  # Read the first line

    values = line.split(',')  # Split the line by commas
    num_values = len(values)

    with open(output_file, 'w') as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")

        # Write header
        ply_file.write(f"element vertex {num_values}\n")  # We're extracting all values
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")

        # Write readings to PLY file
        for i, distance_str in enumerate(values):
            distance = float(distance_str)
            # Calculate x, y, z coordinates assuming a cylindrical coordinate system
            angle = math.radians(i)  # Assuming 0 degrees is the start
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            ply_file.write(f"{x} {y} {distance}\n")  # Using distance as the z-coordinate

# Example usage:
import math

txt_file = 'myfile.txt'  # Path to your input text file
ply_file = 'output.ply'  # Path where you want to save the PLY file

# Convert text file to PLY file
txt_to_ply(txt_file, ply_file)
