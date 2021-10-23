#!/bin/bash

#Enable extglob for inverse globbing
shopt -s extglob

#Parameters
##Defaults
dir="./"
subdir="colorcorrect"
map_file=""
whitebalance_script=""
spec_file="img_rotate_crop.csv"
##Read in parameters
while getopts ":d:s:w:m:c:" opt; do
        case ${opt} in
				d )
						dir=$OPTARG
						;;
				s )
						subdir=$OPTARG
						;;
				w )
						whitebalance_script=$OPTARG
						;;
				m )
						map_file=$OPTARG
						;;
				c )
                        spec_file=$OPTARG
                        ;;
				\? )
						echo "Invalid option: $OPTARG" 1>&2
						;;
				: )
						echo "Invalid option: $OPTARG requires an argument" 1>&2
						;;
        esac
done
shift $((OPTIND -1))

##Validate Parameters
[ -z $whitebalance_script ] && (>&2 echo "ERROR: Need to specify the location of the whitebalance_interpolate.py script."; exit 1)
[ -z $map_file ] && (>&2 echo "ERROR: Need to specify the whitebalance map file."; exit 1)

#Assume the rotation/crop specification file is always in the directory where the base images are.
spec_file="${dir}/${spec_file}"
map_file="${dir}/${map_file}"

#Make the subdirectory where images will be processed
mkdir -p ${dir}/${subdir}

IFS=','
while read file rotation_spec crop_spec output_filename; do
        input_file_full=${dir}/${file}
        output_file_full=${dir}/${subdir}/${file}
		cmd="$whitebalance_script -i ${input_file_full} -o ${output_file_full} -m ${map_file}"
		echo $cmd
        eval $cmd
done < $spec_file
