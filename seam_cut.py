def create_seam_cut(orig_scene, mask_scene,
                    match_scene=None, orig_scene_no_mask=None):

    if match_scene == None:
        match_scene = np.ones(orig_scene.shape) * 255
        match_scene[orig_scene.shape[0], orig_scene.shape[1]] = 0
    if orig_scene_no_mask == None:
        orig_scene_no_mask = np.ones(orig_scene.shape) * 255
        orig_scene_no_mask[orig_scene.shape[0], orig_scene.shape[1]] = 0

    diff = np.absolute(np.subtract(match_scene, orig_scene))
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float)

    for x in range(mask_scene.shape[0]):
        for y in range(mask_scene.shape[1]):
            if mask_scene[x, y] == 0:
                diff_gray[x, y] = np.inf

    mask_info = np.where(mask_scene == 0)

    min_x, max_x, min_y, max_y = min(mask_info[0]), max(mask_info[0]), min(mask_info[1]), max(mask_info[1])

    adj = 10
    dim_diff = diff_gray.shape

    NW_top = zip([0] * len(range(0, min_y - adj)), range(0, min_y - adj))
    NW_left = zip(range(0, min_x - adj), [0] * len(range(0, min_x - adj)))
    top_left = NW_top + NW_left

    NE_top = zip([0] * len(range(max_y + adj, dim_diff[1])), range(max_y + adj, dim_diff[1]))
    NE_right = zip(range(0, min_x - adj), [dim_diff[1] - 1] * len(range(0, min_x - adj)))
    top_right = NE_top + NE_right

    SW_left = zip([dim_diff[0] - 1] * len(range(0, min_y - adj)), range(0, min_y - adj))
    SW_bot = zip(range(max_x + adj, dim_diff[0]), [0] * len(range(max_x + adj, dim_diff[0])))
    bottom_left = SW_left + SW_bot

    SE_right = zip([dim_diff[0] - 1] * len(range(max_y + adj, dim_diff[1])), range(max_y + adj, dim_diff[1]))
    SE_bot = zip(range(max_x + adj, dim_diff[0]), [dim_diff[1] - 1] * len(range(max_x + adj, dim_diff[0])))
    bottom_right = SE_right + SE_bot

    diff_path = np.zeros(diff_gray.shape)

    try:
        costMCP = skimage.graph.MCP(diff_gray, fully_connected=True)
        cumpath, trcb = costMCP.find_costs(starts=NW_left, ends=NE_right)

        for _ in range(10):
            path_tltr = costMCP.traceback(choice(NE_right))  # select a random end point
            for x, y in path_tltr:
                diff_path[x, y] = 255
    except:
        pass

    try:
        costMCP = skimage.graph.MCP(diff_gray, fully_connected=True)
        cumpath, trcb = costMCP.find_costs(starts=NW_top, ends=SW_bot)

        # get 10 random paths...
        for _ in range(10):
            path_tlbl = costMCP.traceback(choice(SW_bot))
            for x, y in path_tlbl:
                diff_path[x, y] = 255
    except:
        pass

    try:
        costMCP = skimage.graph.MCP(diff_gray, fully_connected=True)
        cumpath, trcb = costMCP.find_costs(starts=NE_top, ends=SE_bot)

        # get 10 random paths...
        for _ in range(10):
            path_trbr = costMCP.traceback(choice(SE_bot))
            for x, y in path_trbr:
                diff_path[x, y] = 255
    except:
        pass

    try:
        costMCP = skimage.graph.MCP(diff_gray, fully_connected=True)
        cumpath, trcb = costMCP.find_costs(starts=SW_left, ends=SE_right)

        # get 10 random paths...
        for _ in range(10):
            path_blbr = costMCP.traceback(choice(bottom_right))
            for x, y in path_blbr:
                diff_path[x, y] = 255
    except:
        pass

    h, w = diff_path.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    diff = (1, 1)

    orig_mask = np.where(orig_scene_no_mask == 0)

    for _ in range(10):
        try:
            rnd_point = choice(range(len(orig_mask[0])))
            diff_fill = cv2.floodFill(diff_path.astype(np.uint8), mask,
                                      (orig_mask[0][rnd_point], orig_mask[1][rnd_point]), (255, 255), diff, diff)[1]
            break
        except:
            pass
            
    kernel = np.ones((5, 5), np.uint8)
    diff_fill = cv2.erode(diff_fill, kernel, iterations=2)
    diff_fill = cv2.dilate(diff_fill, kernel, iterations=2)

    diff_fill = cv2.blur(diff_fill, (10, 10))
    diff_fill = cv2.threshold(diff_fill, 5, 255, cv2.THRESH_BINARY)[1]
    return np2to3(diff_fill)