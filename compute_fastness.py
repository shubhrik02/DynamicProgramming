import numpy as np
import cv2
import matplotlib.pyplot as plt

#loading matrix for both videos
data1 = np.load('outputs/ChComboSpin4_n01_p03_g11_3d_output.npy')
data2 = np.load('outputs/ChComboSpin4_n01_p02_g12_3d_output.npy')

#Taking root joint as pelvic
root_position_1 = data1[0][0]
root_position_2 = data2[0][0]

#loading positions for frames that match
frame_indices = [252,253,254,255,256,257,258,263,267,269,270,272,273,274,283]
joint_positions_1 = []
for frame_idx in frame_indices:
    frame_data = data1[frame_idx]
    frame_joint_positions = frame_data[:, :3]  # select only the (x, y) positions of all joints
    joint_positions_1.append(frame_joint_positions)
joint_positions_1 = np.array(joint_positions_1)

#extract positions of joints for player 2
frame_indices2 =[254,255,256,257,258,259,260,265,269,271,272,273,278,279,280]  #matching frames found between video 1 and video2
joint_positions_2 = []
for frame_idx2 in frame_indices2:
    frame_data2 = data2[frame_idx2]
    frame_joint_positions2 = frame_data2[:,:3]
    joint_positions_2.append(frame_joint_positions2)
joint_positions_2 = np.array(joint_positions_2)

# Function to normalize joint positions w.r.t root joint
def normalize_skeletons(skeletons, root):
        # determine the reference joint (root joint)
        ref_joint = root

        # subtract the coordinates of the reference joint from all joint coordinates
        skeletons -= ref_joint

        # subtract the x, y, and z coordinates of the reference joint from all other joint coordinates
        skeletons[:, 1:, :] -= ref_joint

        return skeletons

# Compute average distance between joint positions in consecutive frames
def compute_fastness(joint_positions):
    """
    Compute fastness of player given their joint positions
    Args:
        joint_positions: array of joint positions, shape (num_frames, num_joints, num_dimensions)
    Returns:
        fastness: average distance between joint positions in consecutive frames
    """
    num_frames, num_joints, num_dimensions = joint_positions.shape
    distances = np.sqrt(np.sum(np.square(joint_positions[:-1] - joint_positions[1:]), axis=(1, 2)))
    fastness = np.mean(distances)
    speed = fastness/num_frames



    return fastness, speed

#normalize joint positions using position of one of the joints, here pelvic
normalized_joint_positions_1  = normalize_skeletons(joint_positions_1, root_position_1)
normalized_joint_positions_2 = normalize_skeletons(joint_positions_2, root_position_2)

#print(normalized_joint_positions_1)


#Compute average displacement between joints of first player based on matched frames
fastness_1, speed_1 = compute_fastness(joint_positions_1)
print('Score for player 1 based on matched frames: ', fastness_1)

#Compute average displacement between joints of second player based on matched frames
fastness_2, speed_2 = compute_fastness(joint_positions_2)
print('Score for player 2 based on matched frames:', fastness_2)


#Computer average displacement for player 1 based on entire video
fastness_fullvideo_1, speed_fullvideo1 = compute_fastness(data1)
print('Score for player 1 (entire video):', fastness_fullvideo_1)

#Computer average displacement for player 2 based on entire video
fastness_fullvideo_2, speed_fullvideo2 = compute_fastness(data2)
print('Score for player 2 (entire video):', fastness_fullvideo_2)

#Comparison based on matched frames
if fastness_1 < fastness_2:
     print('Player 1 is faster than player 2 in matched frames')
elif fastness_1 > fastness_2:
    print('Player 2 is faster than player 1 in matched frames')
else:
     print('Both players have the same fastness in matched frames')

#Comparison based on entire video
if fastness_fullvideo_1 < fastness_fullvideo_2:
     print('Player 1 is faster than player 2 in the full video')
elif fastness_fullvideo_1 > fastness_fullvideo_2:
    print('Player 2 is faster than player 1 in the full video')
else:
     print('Both players have the same fastness in the full video')


# load video and deviation values
cap = cv2.VideoCapture('marked_frames/merged_marked_video5_6.mp4')

# Get video dimensions and FPS
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('marked_frames/fastness_annotated_video.mp4', fourcc, fps, (width, height))

# loop over video frames
frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # annotate frame with player fastness
    if frame_num < len(frame_indices) and frame_num < len(frame_indices2):
        if fastness_1 < fastness_2:
            text = "Player 1 is faster than Player 2"
        elif fastness_2 < fastness_1:
            text = "Player 2 is faster than Player 1"
        else:
            text = "Both players have the same fastness"

        # add text overlay on frame
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # write annotated frame to output video
    output_video.write(frame)

    # increment frame counter
    frame_num += 1

# release resources
cap.release()
output_video.release()
cv2.destroyAllWindows()
