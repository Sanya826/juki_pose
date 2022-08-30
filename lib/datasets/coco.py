COCO_PERSON_SKELETON = [[0,1],[1,4],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[7,10],[8,9],[10,11]]#[[2, 3], [3,4], [4, 5], [2, 6], [6,7], [7,8], [2,1]] 
    
COCO_KEYPOINTS = [
    "left_finger_tip",
    "left_finger_base",
    "right_finger_tip",
    "right_finger_base",
    "wrist",
    "elbow",
    "shoulder",
    "body",
    "left_knee",
    "left_foot",
    "right_knee",
    "right_foot"]

HFLIP = {
    'left_finger_tip': 'right_finger_tip',
    'right_finger_tip': 'left_finger_tip',
    'left_finger_base': 'right_finger_base',
    'right_finger_base': 'left_finger_base',
    'left_knee': 'right_knee',
    'right_knee': 'left_knee',
    'left_foot': 'right_foot',
    'right_foot': 'left_foot',
}

COCO_PERSON_SIGMAS = [
    0.026,  # finger_tip
    0.026,  # finger_tip
    0.025,  # finger_base
    0.025,  # finger_base
    0.035,  # wrist
    0.079,  # elbow
    0.079,  # shoulder
    0.072,  # body
    0.072,  # knee
    0.062,  # knee
    0.062,  # foot
    0.062   # foot
]

def print_associations():
    for j1, j2 in COCO_PERSON_SKELETON:
        print(COCO_KEYPOINTS[j1 - 1], '-', COCO_KEYPOINTS[j2 - 1])


if __name__ == '__main__':
    print_associations()
    draw_skeletons()
