MEDIA=../Data/Media

canon_t6_calibrate:
		./calibrate_camera.py --descriptor "Canon EF-S 18-55mm f/3.5-5.6 IS II: 18mm" -d $(MEDIA)/canonT6/EF-S_IS-II_18mm/Checkerboard_Large -o $(MEDIA)/canonT6/EF-S_IS-II_18mm/Checkerboard_Large -s 108

canon_t6_calibrate_circles:
		./calibrate_camera.py -e "JPG" --type circles --descriptor "Canon EF-S 18-55mm f/3.5-5.6 IS II: 18mm" -d $(MEDIA)/canonT6/EF-S_IS-II_18mm/Circles_12mm -o $(MEDIA)/canonT6/EF-S_IS-II_18mm/Circles_12mm -s 12 -x 8 -y 11

dji_p4p_calibrate:
		./calibrate_camera.py -e "JPG" --descriptor "DJI Phantom 4 Pro" -d $(MEDIA)/DJIP4P/FL_24.0mm/Checkerboard_Large -o $(MEDIA)/DJIP4P/FL_24.0mm/Checkerboard_Large -s 108 --draw

dji_p4p_calibrate_circles:
		./calibrate_camera.py -e "JPG" --type circles --descriptor "DJI Phantom 4 Pro" -d $(MEDIA)/DJIP4P/FL_24.0mm/Circles_12mm -o $(MEDIA)/DJIP4P/FL_24.0mm/Circles_12mm -s 12 -x 8 -y 11

undistort:
		./undistort_image.py -v -p $(MEDIA)/canonT6/EF-S_IS-II_18mm/Circles_12mm/calibrate_params.pickle -i $(MEDIA)/canonT6/EF-S_IS-II_18mm/distort.JPG -o $(MEDIA)/canonT6/EF-S_IS-II_18mm/un-distort.JPG
