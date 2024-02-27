# Lane-Detection-Extension

The original code performed Canny edge detection on a road image to try and detect lane lines. However, it had limitations dealing with curved lanes, faded markings, shadows etc. I implemented some improvements like using color thresholds for white/yellow lanes, morphological dilation to connect broken edges, and allowing gaps when fitting lines with Hough transform.

To compare performance, I applied the lane detection to custom images. Potential failure cases were analyzed: curved lanes, fading paint, shadows, road cracks etc. Other edge detectors like Sobel and Laplacian were tried to see if they perform better.

Finally, I created a more robust evaluation pipeline for edge detection on road images. This involved:

Loading custom images
Manually labeling ground truth lanes
Applying Canny and Sobel edge detection
Visually assessing and comparing the outputs
Quantitatively measuring precision and recall metrics

This covers the complete flow - from detecting lanes, to analyzing failure cases, to trying alternate techniques, to systematically benchmarking performance on real images. The ultimate goal was to improve robustness and accuracy of lane/edge detection using both qualitative and quantitative analysis.
