## Opgaver og læsestof uge 1 - De basale teknikker

Filtre, thresholding, farvebehandling, edge detection.

## Artikler, eksempler osv.

Hvis ikke andet er nævnt, så indeholder link både forklaringer og kodeeksempler i både C++ og Python.

### Basale filtre

* Gråtonefilter:
* HSV filtrering (fjern bestemte farver/isolér bestemte farver): [https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html](https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html) 
* Blur filtre: [https://docs.opencv.org/4.x/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html](https://docs.opencv.org/4.x/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html)
* Erotion og dilation (fremhæv linjer/forstærk linjer/reparér brudte linjer): [https://docs.opencv.org/4.x/db/df6/tutorial_erosion_dilatation.html](https://docs.opencv.org/4.x/db/df6/tutorial_erosion_dilatation.html)
    * Avancerede former: [https://docs.opencv.org/4.x/d3/dbe/tutorial_opening_closing_hats.html](https://docs.opencv.org/4.x/d3/dbe/tutorial_opening_closing_hats.html)
* Thresholding (filtrering baseret på grænseværdier): [https://docs.opencv.org/4.x/db/d8e/tutorial_threshold.html](https://docs.opencv.org/4.x/db/d8e/tutorial_threshold.html)

### Basale detektionsalgoritmer

* Sobel (kantdetektering):  [https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html](https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html)
* Canny (kantdetektering; kraftigere men en større proces): [https://docs.opencv.org/4.x/da/d5c/tutorial_canny_detector.html](https://docs.opencv.org/4.x/da/d5c/tutorial_canny_detector.html)
* Hough (cirkeldetektering): [https://docs.opencv.org/4.x/d4/d70/tutorial_hough_circle.html](https://docs.opencv.org/4.x/d4/d70/tutorial_hough_circle.html)
* Hough-Ballard og Hough-Guil detektering. Detekter mere komplicerede objekter, såfremt de indeholder basale geometriske former: [https://docs.opencv.org/4.x/da/ddc/tutorial_generalized_hough_ballard_guil.html](https://docs.opencv.org/4.x/da/ddc/tutorial_generalized_hough_ballard_guil.html)
* Contour detection (find omrids; brug sammen med erotion og dilation for bedste resultater): [https://docs.opencv.org/4.x/df/d0d/tutorial_find_contours.html](https://docs.opencv.org/4.x/df/d0d/tutorial_find_contours.html) 
* Blob detection (find mere avancerede former, mange indstillinger): [https://learnopencv.com/blob-detection-using-opencv-python-c/](https://learnopencv.com/blob-detection-using-opencv-python-c/)

### Andre

* Histogram equalization (kontrastbalancering): [https://docs.opencv.org/4.x/d4/d1b/tutorial_histogram_equalization.html](https://docs.opencv.org/4.x/d4/d1b/tutorial_histogram_equalization.html)
* Template matching (mest basale recognition model; find hvad der ligner eksempelbillede): [https://docs.opencv.org/4.x/de/da9/tutorial_template_matching.html](https://docs.opencv.org/4.x/de/da9/tutorial_template_matching.html)

## Opgaver og læsestof uge 2 - Fokus på detektering og segmentering

## Artikler eksempler osv.

### Segmentering 

* Distance transform og watershed (god baseline/start): [https://docs.opencv.org/4.x/d2/dbd/tutorial_distance_transform.html](https://docs.opencv.org/4.x/d2/dbd/tutorial_distance_transform.html)
* Anisotropic segmentering (avanceret at forstå, men ikke avanceret at prøve): [https://docs.opencv.org/4.x/d4/d70/tutorial_anisotropic_image_segmentation_by_a_gst.html](https://docs.opencv.org/4.x/d4/d70/tutorial_anisotropic_image_segmentation_by_a_gst.html)

### Detektering 

Basal

* Harris corner (hjørnedetektering): [https://docs.opencv.org/4.x/d4/d7d/tutorial_harris_detector.html](https://docs.opencv.org/4.x/d4/d7d/tutorial_harris_detector.html) 
    * Alternativt Shi-Tomasi: [https://docs.opencv.org/4.x/d8/dd8/tutorial_good_features_to_track.html](https://docs.opencv.org/4.x/d8/dd8/tutorial_good_features_to_track.html)
* Image moments (kan bruges til analyse af konturer): [https://docs.opencv.org/4.x/d0/d49/tutorial_moments.html](https://docs.opencv.org/4.x/d0/d49/tutorial_moments.html)

Avanceret

* Feature detection via ORB, SIFT, SURF: [https://docs.opencv.org/4.x/d7/d66/tutorial_feature_detection.html](https://docs.opencv.org/4.x/d7/d66/tutorial_feature_detection.html)
* FLANN for feature matching (objektdetektering): [https://docs.opencv.org/4.x/d5/d6f/tutorial_feature_flann_matcher.html](https://docs.opencv.org/4.x/d5/d6f/tutorial_feature_flann_matcher.html)
    * Alternativt AKAZE: [https://docs.opencv.org/4.x/db/d70/tutorial_akaze_matching.html](https://docs.opencv.org/4.x/db/d70/tutorial_akaze_matching.html)

### Machine Learning 

Machine learning er ret avanceret og hardwaretungt. Derfor er det ikke obligatorisk, men det er anbefalelsesværdigt at prøve . Det er anvendeligt uanset use case, men forskellige modeller til forskellige use cases. OBS: Kan være langsomt hvis i ikke har en Nvidia GPU.

* Definitiv DNN guide: [https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/](https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/)
    
