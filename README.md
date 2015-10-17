# Tilt Shift Generator
Using OpenCV, generate lens blurred images and merge with mask and original image together with a
blurred version of the image to produce fake tilt-shift photos.

## Background
In the CS64675 Computational Photography course the final project is entirely open ended. Students
are allowed to pick a particular topic of interest, propose that topic, and then spend the remaining
weeks of the semester researching and developing some final artifact.

The author has a great fondness of tilt-shift photography. The effect that is produced on images
made with DSLR or even traditional cameras, is incredibly unique and adds an amazing character
to images. Due to this interest the author decided to focus on using Python and OpenCV, both used
heavily throughout the course, to produce a generator of sorts that could produce these types of
images.

## Objective
The goal of this project was to create fake tilt-shift photos. The project reuses code from a
previous course assignment that merged two images together according to a mask. The author reused
this code and then focused on how to fake the tilt-shift effect with various blurs. The final
aspect was determine what made up a good mask to merge the two images together. The author used
real tilt-shift images as a basis for evaluation of the tool's ability to produce the tilt-shift
effect.


## Getting Started
The primary script is run_all.py. It will run both the blurring of an image as well as the blending
of the two images together with a provided mask. Here is a quick run down of the steps required:

1. Get a source image
2. Produce a greyscale mask of the image:
  * White areas become blurred
  * Black areas become left in the original state
  * Grey areas become a mixture of blurred depending on how white the intensity it
  * The mask must be named the same as the input image.
3. Place the source image in the *images/original* folder
4. Place the mask image in the *images/mask* folder
5. Execute *run_all.py*
6. This will produce a blurred version of the image, merge the images according to the mask, and
output the final image into the *images/output* folder

## Directory Structure
The folder structure should look like:
```
src/images
 |
 +-- original
 |  |
 |  \-- image.jpeg
 |
 +-- mask
 |  |
 |  \-- image.jpeg
```

## Technologies Used
 * **Python**
   * The author's favorite programming language
   * Also the primary language used for the course
 * **OpenCV**
   * A fantastic computer vision library
   * Find more information at: http://www.opencv.org

## Contributors
* [Joshua Powers](http://powersj.github.io/)
  * CS6475 Computational Photography (Summer 2015)
  * Georgia Institute of Technology

## License
MIT &copy; Joshua Powers
