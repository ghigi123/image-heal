import numpy as np
import cv2
from skimage.graph import MCP
from random import choice
import itertools

n_success = itertools.count()
n_errors = itertools.count()

def np2to3(im):
    # convert 2d to 3d naively
    new_im = np.zeros((im.shape[0], im.shape[1], 3))
    r, c = im.shape
    for x in range(r):
        for y in range(c):
            new_im[x, y, :] = im[x, y]
    return new_im

def traceback_mcp(mcp, starts, ends):
    if starts and ends:
        cumpath, trcb = mcp.find_costs(starts=starts, ends=ends)
        print(trcb)
        print(cumpath)
        for end in ends:
            try:
                for start in mcp.traceback(end):
                    yield start
                print('success', next(n_success))
            except:
                print('errors', next(n_errors))


def create_seam_cut(orig_scene, mask_scene, match_scene):
    diff = np.absolute(match_scene - orig_scene).numpy()

    diff_gray = 0.299 * diff[0] + 0.587 * diff[1] + 0.114 * diff[2]

    diff_gray[mask_scene == 0] = np.inf

    mask_info = np.where(mask_scene == 0)

    min_y = 20
    adj = 10
    height, width = diff_gray.shape

    NW_top = [(0, j) for j in range(0, min_y - adj)]
    NW_left = [(i, 0) for i in range(0, min_x - adj)]

    NE_top = [(0, j) for j in range(max_y + adj, width)]
    NE_right = [(i, width - 1) for i in range(0, min_x - adj)]

    SW_left = [(height - 1, j) for j in range(0, min_y - adj)]
    SW_bot = [(i, 0) for i in range(max_x + adj, height)]

    SE_right = [(height - 1, j) for j in range(max_y + adj, width)]
    SE_bot = [(i, width - 1) for i in range(max_x + adj, height)]

    diff_path = np.zeros(diff_gray.shape)


    mcp = MCP(diff_gray, fully_connected=True)

    for position in traceback_mcp(mcp, NW_left, NE_right):
        diff_path[position] = 1

    for position in traceback_mcp(mcp, NW_top, SW_bot):
        diff_path[position] = 1

    for position in traceback_mcp(mcp, NE_top, SE_bot):
        diff_path[position] = 1

    for position in traceback_mcp(mcp, SW_left, SE_right):
        diff_path[position] = 1

    mask = np.zeros((height + 2, width + 2), np.uint8)
    diff = (1, 1)

    orig_mask = np.where(orig_scene == 0)

    for _ in range(10):
        try:
            rnd_point = choice(range(width))
            diff_fill = cv2.floodFill(diff_path.astype(np.uint8), mask,
                                      (orig_mask[1][rnd_point], orig_mask[2][rnd_point]), (1, 1), diff, diff)[1]
            break
        except:
            pass
            
    kernel = np.ones((5, 5), np.uint8)
    diff_fill = cv2.erode(diff_fill, kernel, iterations=2)
    diff_fill = cv2.dilate(diff_fill, kernel, iterations=2)

    diff_fill = cv2.blur(diff_fill, (10, 10))
    diff_fill = cv2.threshold(diff_fill, 5, 255, cv2.THRESH_BINARY)[1]
    return np2to3(diff_fill)

def create_seam_cut(orig_scene, mask_scene, match_scene):
    _, height, width = orig_scene.shape

    mask_info = np.where(mask_scene == 0)
    min_x, max_x, min_y, max_y = min(mask_info[1]), max(mask_info[1]), min(mask_info[2]), max(mask_info[2])

    match_scene = match_scene[min_x:max_x, min_y:max_y]

    return cv2.seamlessClone(match_scene.numpy(), orig_scene.numpy(), mask_scene.numpy(), (min_x + (max_x - min_x) // 2, min_y + (max_y - min_y) // 2), cv2.NORMAL_CLONE)

if __name__ == '__main__':
    import torch
    fake_image = torch.Tensor([[[i + j + k for j in range(20)] for i in range(20)] for k in range(3)])

    print(create_seam_cut(fake_image, torch.Tensor([[[i + j > 5 for j in range(20)] for i in range(20)] for k in range(3)]), fake_image))
