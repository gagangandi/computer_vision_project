def estimate_distance(width_px, mtx, real_width_cm=180, f_mm=4.73,
                      video_resolution=[1920, 1080], image_resolution=[2608, 4624]):

    fx = mtx[0][0]
    fy = mtx[1][1]
    f_avg = (fx + fy) / 2
    m = f_avg / f_mm
    new_m = (video_resolution[0] * m) / image_resolution[0]

    distance = (real_width_cm * f_mm * new_m) / width_px
    return distance
