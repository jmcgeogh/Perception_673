This README file will cover the steps to run the three files associated with Project 1. The files for part 2 have the same steps.

project1_673_part1.py:

1) This file runs with the "Tag1'' video. Ensure that the video has been downloaded before running this program.
2) The libraries used in this program are cv2, numpy, matplotlib.pyplot, and scipy. Ensure that all libraries have been downloaded prior to running this program.
3) Part 1b of this program uses the refrance_marker.png file. The file has been included in the zip folder, ensure it is outside the zip before running this program.
4) Run the program
5) Part 1a will run and you will have output images displayed on the screen using frame 100 of the Tag1 video. All output images need a key input to close and have the program continue running. Each image will also be save to the location of the folder where the program is being run.
6) Once part 1a of the program has finished running part 1b will start automatically
7) The user will be prompted to select if they want to run part 1b with the reference image or the frame from part 1a.
8) Once selected the user will then be prompted to flip the image if they wish to
9) If 'n' is selected skip to step 12
10) If 'y' is selected the user will then be prompted for how they would like to flip the image
11) Once the flip type has been selected the program will continue running
12) The program will then output two images, both needing key input to close and move on. Both will also be saved in the location that the program is being run from.
13) The program will then close
14) Return to step 4

project1_673_part2a.py and project1_673_part2a.py:

1) 1) This file runs with the "Tag0", "Tag1", "Tag2", and "Multiple_tags'' videos. Ensure that the video has been downloaded before running this program.
2) The libraries used in this program are cv2, numpy, copy, math, and scipy. Ensure that all libraries have been downloaded prior to running this program.
3) The program has commented out functions and sections for video creation. All sections for video creation are labeled as such. If you do not wish to create a video output skip to step 8
4) Uncomment all sections marked for video create
5) Create a frame storage folder, this is where the program will put the frames created for the video
6) Edit the variable called 'path' with the location of the frame storage folder
7) In the visual function, rename the video to whatever name you would like the output to be called
8) Run the program
9) The user will be prompted to select the video file that they would like to run
10) Once the video has been selected the console will output the frame count of the video
11) The original frame, warped tag, and final frame will then all be displayed and updated as the program runs through the selected video
12) If the user decided to visualize the output then the program will create the video in the same folder as the program was run before closing, otherwise the program will close
13) Return to step 3
