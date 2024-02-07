# Python Script to fly in AirSim
# Python 3.8

import airsim  # 1.8.1
import numpy as np  # 1.24.4
import PIL  # 10.1.0
import pyrealsense2 as rs
import os
from pprint import pprint
from PIL import Image, ImageOps, ImageGrab
print(airsim.__version__)
print(np.__version__)
print(PIL.__version__)

save_screenshot = None


def get_position(client_drone):
    """
    Extracts the x, y and z from the drone-position-vector into 3 seperate floats.

    Args:
    - client_drone (airsim.MultirotorClient): the AirSim client you are flying with. You can create one like the following: client = airsim.MultirotorClient()

    Returns:
    - x (float)
    - y (float)
    - z (float)
    """

    # Get the state of the drone
    state = client_drone.getMultirotorState()

    # Get the position of the drone
    position = state.kinematics_estimated.position

    # Return Pos
    return position.x_val, position.y_val, position.z_val


def get_euler_angles(client_drone):
    """
    Extracts the pitch, roll and yaw from the drone-pose-vector into 3 separate floats.

    Args:
    - client_drone (airsim.MultirotorClient): the AirSim client you are flying with. You can create one like the following: client = airsim.MultirotorClient()

    Returns:
    - pitch (float)
    - roll (float)
    - yaw (floats)
    """
    # Get the pose of the drone
    pose = client_drone.simGetVehiclePose()

    # Extract pitch, roll, and yaw from the pose
    pitch, roll, yaw = airsim.to_eularian_angles(pose.orientation)

    # Return Values
    return pitch, roll, yaw


def get_velocity(client_drone):
    """
    Logic from Sieuwe to calculate velocity

    Args:
    - client_drone (airsim.MultirotorClient): the AirSim client you are flying with. You can create one like the following: client = airsim.MultirotorClient()

    Returns:
    - velocity (float)

    """
    gps_data = client_drone.getGpsData()
    # ??? - ask Sieuwe
    velocity = gps_data.gnss.velocity.x_val**2+gps_data.gnss.velocity.y_val**2
    return velocity


def get_screenshot():
    """
    Takes a screenshot of the screen

    Returns: 
    - screenshot (PIL.Image)"""
    # take screenshot
    screenshot = ImageGrab.grab()
    # save image
    screenshot.save("test.png")
    # return a Pil image
    return screenshot


def get_img_state(image, grid_size, threshold):
    """Returns an array with 1s and 0s representing black and white gridfiles. black=0"""
    # Grayscale image
    gray_image = ImageOps.grayscale(image)

    # Region prep
    width, height = gray_image.size
    image_array = np.array(gray_image)
    height, width = image_array.shape
    region_size = (height // grid_size, width // grid_size, 1)

    # Split the image into IxJ regions and extract only the red channel
    regions = [image_array[i * region_size[0]: (i + 1) * region_size[0],
                           j * region_size[1]: (j + 1) * region_size[1]]
               for i in range(grid_size) for j in range(grid_size)]

    # convert to np array
    regions_array = np.array(regions)

    # maximize and minimize values
    state = []
    for region in regions_array:
        region_value = np.min(region)
        # print(region_value)
        if region_value < threshold:
            state.append(1)
        else:
            state.append(0)
    return state


def get_depth_cam_pov():
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 2)


# add - x, y, z, deltas -> to understand where to fly to
def get_state():
    """
    Returns:
    - state (list): list containing up to date variables of the environment.
        - x (float)
        - y (float)
        - z (float)
        - pitch (float)
        - roll (float)
        - yaw (float)
        - velocity (float)
        - img_state (list): list with 0s and 1s representing the drone's pov
    """
    grid_size = 5
    threshold = 60

    img_state = get_img_state(pil_image, grid_size, threshold)
    x, y, z = get_position(client)
    pitch, roll, yaw = get_euler_angles(client)
    velocity = get_velocity(client)

    return [x, y, z, pitch, roll, yaw, velocity, img_state]

# --------------
# SET CONNECTION
# --------------


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# ------------
# START FLYING
# ------------

# Async methods returns Future. Call join() to wait for task to complete.
client.takeoffAsync().join()

# code to keep flying forward
# for i in range(5):
#     # keep flying forward
#     pos = get_position(client)
#     client.moveToPositionAsync(pos.x_val+10, pos.y_val, pos.z_val, 10).join()

print('start loop')
while True:
    # TAKE IMAGE
    responses = client.simGetImages(
        [airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False)])
    depth_image = responses[0]
    arr = airsim.list_to_2d_float_array(
        depth_image.image_data_float, depth_image.width, depth_image.height)
    pil_image = Image.fromarray(arr)

    # GET STATE
    state = get_state()
    print('STATE')
    print(state)

    # GET ACTION

    # rpt
    break


# ------------
# ????????????
# ------------
