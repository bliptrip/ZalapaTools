#This Makefile is mostly intended to demonstrate usage of the scripts embedded within this tool.
DEBUGP= 
ifneq ($(DEBUG),)
		DEBUGP=python3 -m pdb
endif

genref:
	$(DEBUGP) ./genRefCC.py -c ../Data/xrite_cc_colors.csv -o ../Data/xrite_cc_ref_swatch.PNG

sse2Mask:
	$(DEBUGP) ./sse2CCMasks.py --settings ~/Zalapa/projects/Drone/Software/bliptrip/semantic-segmentation-editor/settings.json

mask2CC:
	$(DEBUGP) ./mask2CC.py --settings ~/Zalapa/projects/Drone/Software/bliptrip/semantic-segmentation-editor/settings.json

xrite_cc_rgb2hex:
		./whitebalance_map_rgb2hex.py -m ../Data/xrite_cc_colors.csv

xrite_cc_example:
		./cc_colormap_generator.py -j ../Data/example_xrite_cc.json -i ../Data/example_xrite_cc.JPG -o ../Data/xrite_cc_colormap.csv -c ../Data/xrite_cc_colors.csv -d

xrite_cc_example_2:
		./cc_colormap_generator.py -j ../Data/example_xrite_cc_2.json -i ../Data/example_xrite_cc_2.JPG -o ../Data/xrite_cc_colormap_2.csv -c ../Data/xrite_cc_colors.csv -d

model_generate_example:
		./cc_model_generator.py -m ../Data/xrite_cc_colormap.csv -o ../Data/example_xrite_cc.models -d -i ../Data/example_xrite_cc.includes

model_generate_example_2:
		./cc_model_generator.py -m ../Data/xrite_cc_colormap_2.csv -o ../Data/example_xrite_cc_2.models -d -i ../Data/example_xrite_cc_2.includes

normalize_example:
		./cc_colors_normalize.py -m ../Data/example_xrite_cc.models -i ../Data/example_xrite_cc.JPG -o ../Data/example_xrite_cc.norm.JPG  -d

normalize_example_2:
		./cc_colors_normalize.py -m ../Data/example_xrite_cc_2.models -i ../Data/example_xrite_cc_2.JPG -o ../Data/example_xrite_cc_2.norm.JPG  -d
