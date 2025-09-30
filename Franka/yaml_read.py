import yaml
import os

def read_joint_limits(filename):
    meshdir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(meshdir, filename)
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    
    # Initialize an empty list to store joint limits
    joint_limits_array = []
    
    for joint_name, joint_data in data.items():
        lower_limit = float(joint_data['limit']['lower'])
        upper_limit = float(joint_data['limit']['upper'])
        # Append only the limits as floats [lower_limit, upper_limit]
        joint_limits_array.append([lower_limit, upper_limit])

    # Print the entire 2D array for verification
    print("Joint Limits Array:")
    for row in joint_limits_array:
        print(row)
    
    return joint_limits_array

def read_pose_parameters(filename):
    meshdir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(meshdir, filename)
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    
    # Initialize an empty list to store pose parameters
    pose_parameters_array = []
    
    for joint_name, joint_data in data.items():
        x = float(joint_data['kinematic']['x'])
        y = float(joint_data['kinematic']['y'])
        z = float(joint_data['kinematic']['z'])
        roll = float(joint_data['kinematic']['roll'])
        pitch = float(joint_data['kinematic']['pitch'])
        yaw = float(joint_data['kinematic']['yaw'])
        # Append pose parameters as floats [x, y, z, roll, pitch, yaw]
        pose_parameters_array.append([x, y, z, roll, pitch, yaw])

    # Print the entire 2D array for verification
    print("Pose Parameters Array (x, y, z, roll, pitch, yaw):")
    for row in pose_parameters_array:
        print(row)
    
    return pose_parameters_array

if __name__ == "__main__":
    limits_array = read_joint_limits("joint_limits.yaml")
    pose_array = read_pose_parameters("kinematics.yaml")