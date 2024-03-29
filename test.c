double in2col(double img[][][], h_filter, w_filter, stride, pad)
{
    int n = sizeof(img) / sizeof(img[0]);
    int H = sizeof(img[0]) / sizeof(img[0][0]);
    int W = sizeof(img[0][0]) / sizeof(img[0][0][0]);
    h_out = (H + 2*pad - h_filter)/stride + 1
    w_out = (H + 2*pad - w_filter)/stride + 1
    img = np.pad(img, [(0, 0), (pad, pad), (pad, pad)], 'constant')

    ancol = np.zeros((h_out*w_out, h_filter * w_filter))

    for y_origin in range(h_out):
        for x_origin in range(w_out):
            for y in range(h_filter):
                for x in range(w_filter):
                    ancol[h_out*y_origin + x_origin, y * h_filter +
                          x] = img[y_origin * stride + y, x_origin * stride + x]
    return ancol
}