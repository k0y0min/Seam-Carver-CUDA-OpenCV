# OpenCV with CUDA for Seam Carving

## Installation

To get started with OpenCV and CUDA for Seam Carving, please follow the installation instructions provided in the [4th and 5th video of this YouTube playlist](https://www.youtube.com/watch?v=-GY2gT2umpk&list=PLkmvobsnE0GHmLeVETd6zbbJSDZJWa5Fw&index=4). No dependencies are required for the python version.

## Running the Application

### C++ Version

1. Ensure you have OpenCV with CUDA installed as per the installation instructions.
2. Open a command line interface.
3. Run the following command to execute the Seam Carver program:

   ```
   seamcarver.exe dimV dimH dir [num] [extension]
   ```

   - `dimV` specifies the number of seams to remove vertically.
   - `dimH` specifies the number of seams to remove horizontally.
   - `dir` denotes the folder containing the images.
   - `num` (optional) indicates the number of images to process.
   - `extension` (optional) represents the file extension (e.g., `.png`).

### C++ Short Demo

- For a short demonstration, please watch this [YouTube video](https://www.youtube.com/watch?v=IvKc6A7mTRc).

### Python Version

1. Open a command line interface.
2. Run the following command to execute the Python version of Seam Carver:

   ```
   python seam_carver.py
   ```

   - This version does not require any command-line arguments but needs an `InputImage.jpg` in the current directory.

### Batch Python Version

1. Open a command line interface.
2. Run the following command to execute the batch Python version of Seam Carver:

   ```
   python batch_seam_carver.py
   ```

   - This version also does not require any command-line arguments but needs an `images/` folder in the current directory.

---

Feel free to update and expand this README as needed, providing additional details or clarifications about your project.
